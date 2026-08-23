#!/usr/bin/env python3
"""Small terminal monitor for llama-swap + llama-server.

The llama-swap endpoint tells us which model is running.  llama-server's
/metrics and /slots endpoints provide request state, token counts, and rates.
No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, ProxyHandler, build_opener


METRIC_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)(?:\s+\d+)?$"
)
ANSI_ESCAPE = re.compile(r"\033\[[0-9;]*m")


def fetch_text(url: str, timeout: float) -> str:
    # Do not accidentally send loopback requests through an HTTP proxy.
    opener = build_opener(ProxyHandler({}))
    request = Request(url, headers={"Accept": "application/json, text/plain"})
    with opener.open(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def fetch_json(url: str, timeout: float) -> Any:
    return json.loads(fetch_text(url, timeout))


def parse_metrics(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in text.splitlines():
        match = METRIC_LINE.match(line.strip())
        if not match:
            continue
        try:
            result[match.group(1)] = float(match.group(2))
        except ValueError:
            pass
    return result


def number(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def fmt_int(value: int | float | None) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def fmt_rate(value: float | None) -> str:
    return "-" if value is None else f"{value:7.1f} t/s"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def decoded_tokens(slot: dict[str, Any]) -> int:
    next_token = slot.get("next_token") or {}
    # llama-server versions have returned both an object and a one-element
    # list for this field.
    if isinstance(next_token, list):
        next_token = next_token[0] if next_token else {}
    if not isinstance(next_token, dict):
        return 0
    return number(next_token.get("n_decoded"))


@dataclass
class Turn:
    """One llama-server inference request (an agent 'turn')."""

    task_id: str
    started: float
    prompt_tokens: int = 0
    prompt_cached: int = 0
    prompt_processed: int = 0
    generated: int = 0
    active: bool = True
    finished: float | None = None

    def update(self, slot: dict[str, Any]) -> None:
        self.prompt_tokens = number(slot.get("n_prompt_tokens"), self.prompt_tokens)
        self.prompt_cached = number(slot.get("n_prompt_tokens_cache"), self.prompt_cached)
        self.prompt_processed = number(
            slot.get("n_prompt_tokens_processed"), self.prompt_processed
        )
        self.generated = decoded_tokens(slot) or self.generated
        self.active = bool(slot.get("is_processing"))
        if not self.active and self.finished is None:
            self.finished = time.monotonic()

    @property
    def elapsed(self) -> float:
        return max(0.0, (self.finished or time.monotonic()) - self.started)


@dataclass
class Monitor:
    server_url: str
    swap_url: str
    timeout: float = 1.5
    previous_metrics: dict[str, float] = field(default_factory=dict)
    previous_poll: float | None = None
    active_turn: Turn | None = None
    last_turn: Turn | None = None
    completed_turns: deque[Turn] = field(default_factory=lambda: deque(maxlen=5))
    last_errors: deque[str] = field(default_factory=lambda: deque(maxlen=3))
    last_rates: dict[str, float] = field(default_factory=dict)
    previous_slot_task: str | None = None
    previous_slot_prompt: int | None = None
    previous_slot_generated: int | None = None
    previous_slot_poll: float | None = None

    def get(self, path: str, base_url: str | None = None) -> Any:
        return fetch_json(f"{(base_url or self.server_url).rstrip('/')}{path}", self.timeout)

    def record_error(self, message: str) -> None:
        if message not in self.last_errors:
            self.last_errors.append(message)

    def poll(self) -> dict[str, Any]:
        now = time.monotonic()
        metrics: dict[str, float] = {}
        slots: list[dict[str, Any]] = []
        running: list[dict[str, Any]] = []
        health: dict[str, Any] | None = None
        swap_ok = False
        server_ok = False

        try:
            running_data = self.get("/running", self.swap_url)
            running = running_data.get("running", [])
            swap_ok = True
        except Exception as exc:  # The dashboard should keep running if a service restarts.
            self.record_error(f"llama-swap: {short_error(exc)}")

        try:
            health = self.get("/health")
            server_ok = health.get("status") == "ok"
        except Exception as exc:
            # An unloaded model can make the backend unavailable; that is handled below.
            self.record_error(f"llama-server health: {short_error(exc)}")

        try:
            metrics = parse_metrics(fetch_text(f"{self.server_url.rstrip('/')}/metrics", self.timeout))
        except Exception as exc:
            self.record_error(f"llama-server metrics: {short_error(exc)}")

        try:
            slots_data = self.get("/slots")
            if isinstance(slots_data, list):
                slots = [slot for slot in slots_data if isinstance(slot, dict)]
        except Exception as exc:
            self.record_error(f"llama-server slots: {short_error(exc)}")

        slot = next((item for item in slots if item.get("is_processing")), None)
        if slot is not None:
            task_id = str(slot.get("id_task", "unknown"))
            if self.active_turn is None or self.active_turn.task_id != task_id:
                self.finish_active_turn()
                self.active_turn = Turn(task_id=task_id, started=now)
            self.active_turn.update(slot)
        elif self.active_turn is not None:
            self.active_turn.active = False
            self.active_turn.finished = self.active_turn.finished or now
            self.finish_active_turn()

        interval = None if self.previous_poll is None else max(0.001, now - self.previous_poll)
        rates: dict[str, float | None] = {}
        rate_sources: dict[str, str] = {}
        slot_live_rates: dict[str, float] = {}
        if slot is not None:
            task_id = str(slot.get("id_task", "unknown"))
            prompt_processed = number(slot.get("n_prompt_tokens_processed"))
            generated = decoded_tokens(slot)
            slot_interval = (
                None
                if self.previous_slot_poll is None
                else max(0.001, now - self.previous_slot_poll)
            )
            if (
                slot_interval is not None
                and task_id == self.previous_slot_task
                and self.previous_slot_prompt is not None
                and prompt_processed >= self.previous_slot_prompt
            ):
                prompt_rate = (prompt_processed - self.previous_slot_prompt) / slot_interval
                if prompt_rate > 0:
                    slot_live_rates["prompt"] = prompt_rate
            if (
                slot_interval is not None
                and task_id == self.previous_slot_task
                and self.previous_slot_generated is not None
                and generated >= self.previous_slot_generated
            ):
                generated_rate = (generated - self.previous_slot_generated) / slot_interval
                if generated_rate > 0:
                    slot_live_rates["generate"] = generated_rate
            self.previous_slot_task = task_id
            self.previous_slot_prompt = prompt_processed
            self.previous_slot_generated = generated
            self.previous_slot_poll = now
        else:
            self.previous_slot_task = None
            self.previous_slot_prompt = None
            self.previous_slot_generated = None
            self.previous_slot_poll = None

        # Use counter deltas for a responsive current rate while the server's gauges
        # remain useful as a longer-running average.  If neither changes during a
        # poll, retain the most recent positive rate instead of displaying 0.0.
        for label, metric_name, gauge_name in (
            ("prompt", "llamacpp:prompt_tokens_total", "llamacpp:prompt_tokens_seconds"),
            ("generate", "llamacpp:tokens_predicted_total", "llamacpp:predicted_tokens_seconds"),
        ):
            gauge = metrics.get(gauge_name)
            if gauge is not None and gauge > 0:
                rates[label] = gauge
                rate_sources[label] = "server avg"
                self.last_rates[label] = gauge

            if interval is not None:
                current = metrics.get(metric_name)
                previous = self.previous_metrics.get(metric_name)
                if current is not None and previous is not None and current >= previous:
                    delta_rate = (current - previous) / interval
                    if delta_rate > 0:
                        rates[label] = delta_rate
                        rate_sources[label] = "live"
                        self.last_rates[label] = delta_rate

            if label in slot_live_rates:
                rates[label] = slot_live_rates[label]
                rate_sources[label] = "slot live"
                self.last_rates[label] = slot_live_rates[label]

            if label not in rates:
                rates[label] = self.last_rates.get(label)
                if rates[label] is not None:
                    rate_sources[label] = "last known"

        self.previous_metrics = metrics
        self.previous_poll = now

        model = running[0] if running else {}
        model_state = model.get("state", "unloaded")
        if not swap_ok:
            overall = "ERROR"
        elif model_state not in ("ready", "loading", "starting"):
            overall = "NOMINAL (no model loaded)"
        elif not server_ok and model_state == "ready":
            overall = "ERROR"
        else:
            overall = "NOMINAL"

        return {
            "now": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "rates": rates,
            "rate_sources": rate_sources,
            "model": model,
            "model_state": model_state,
            "server_ok": server_ok,
            "swap_ok": swap_ok,
            "overall": overall,
        }

    def finish_active_turn(self) -> None:
        if self.active_turn is None:
            return
        if self.active_turn.finished is None:
            self.active_turn.finished = time.monotonic()
        self.active_turn.active = False
        self.last_turn = self.active_turn
        self.completed_turns.appendleft(self.active_turn)
        self.active_turn = None


def short_error(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, URLError):
        return "connection refused/unavailable"
    message = str(exc).strip().replace("\n", " ")
    return message[:100] or type(exc).__name__


def render(monitor: Monitor, snapshot: dict[str, Any], color: bool = True) -> str:
    metrics = snapshot["metrics"]
    rates = snapshot["rates"]
    rate_sources = snapshot["rate_sources"]
    model = snapshot["model"]

    def paint(code: str, value: str) -> str:
        return f"\033[{code}m{value}\033[0m" if color else value

    overall = snapshot["overall"]
    overall_display = paint("32", overall) if overall.startswith("NOMINAL") else paint("31", overall)
    model_name = model.get("model", "-")
    model_state = model.get("state", snapshot["model_state"])

    processing = number(metrics.get("llamacpp:requests_processing"))
    deferred = number(metrics.get("llamacpp:requests_deferred"))
    if monitor.active_turn is not None:
        phase = "PROMPT" if monitor.active_turn.prompt_processed < monitor.active_turn.prompt_tokens else "GENERATING"
        activity = paint("36", phase)
    else:
        activity = "IDLE"

    uncached_input = number(metrics.get("llamacpp:prompt_tokens_total"))
    cached_input = number(metrics.get("llamacpp:prompt_tokens_cached_total"))
    output = number(metrics.get("llamacpp:tokens_predicted_total"))
    total_input = uncached_input + cached_input

    def fit(text: str, width: int) -> str:
        visible = ANSI_ESCAPE.sub("", text)
        if len(visible) > width:
            return visible[: max(0, width - 1)] + "…"
        return text + " " * (width - len(visible))

    terminal_width = shutil.get_terminal_size((96, 24)).columns
    box_width = max(76, min(110, terminal_width))
    content_width = box_width - 4

    def row(text: str = "") -> str:
        return f"│ {fit(text, content_width)} │"

    def kv(label: str, value: str) -> str:
        return f"{label:<12} {value}"

    def columns(left: str, right: str) -> str:
        gap = 4
        left_width = (content_width - gap) // 2
        right_width = content_width - gap - left_width
        return f"│ {fit(left, left_width)}{' ' * gap}{fit(right, right_width)} │"

    def section(title: str) -> str:
        label = f"├─ {title} "
        return label + "─" * max(0, box_width - len(label) - 1) + "┤"

    def rate_line(label: str, key: str) -> str:
        source = rate_sources.get(key, "-")
        return kv(label, f"{fmt_rate(rates.get(key))}  [{source}]")

    lines = [
        "╭" + "─" * (box_width - 2) + "╮",
        row(paint("1;36", "LLAMA MONITOR") + f"   {snapshot['now']}"),
        section("STATUS"),
        columns(kv("overall", overall_display), kv("model", model_name)),
        columns(
            kv("swap", "OK" if snapshot["swap_ok"] else "ERROR"),
            kv("server", "OK" if snapshot["server_ok"] else "ERROR"),
        ),
        columns(kv("state", model_state), kv("requests", f"{processing} active / {deferred} deferred")),
        section("REQUEST"),
        columns(kv("activity", activity), kv("phase", phase if monitor.active_turn else "IDLE")),
        section("THROUGHPUT  (positive rate retained when the live counter is idle)"),
        columns(rate_line("prompt", "prompt"), rate_line("generation", "generate")),
        section("SERVER TOTALS  (current llama-server process)"),
        columns(kv("input", f"{fmt_int(total_input)} total"), kv("output", f"{fmt_int(output)} generated")),
        columns(kv("uncached", fmt_int(uncached_input)), kv("cached", fmt_int(cached_input))),
        row(kv("all tokens", fmt_int(total_input + output))),
    ]

    turn = monitor.active_turn or monitor.last_turn
    if turn is None:
        lines.extend([section("CURRENT TURN"), row("none")])
    else:
        label = "active" if turn.active else "last completed"
        lines.extend([
            section(f"CURRENT TURN  [{label}]"),
            columns(kv("task", turn.task_id), kv("elapsed", fmt_duration(turn.elapsed))),
            columns(kv("phase", phase if turn.active else "DONE"), kv("input", f"{fmt_int(turn.prompt_tokens)} total")),
            columns(kv("cached", fmt_int(turn.prompt_cached)), kv("output", f"{fmt_int(turn.generated)} generated")),
            row(kv("turn total", fmt_int(turn.prompt_tokens + turn.generated))),
        ])

    if monitor.completed_turns:
        lines.append(section("RECENT COMPLETED TURNS"))
        for item in list(monitor.completed_turns)[:3]:
            lines.append(columns(
                kv("task", item.task_id),
                kv("input", fmt_int(item.prompt_tokens)),
            ))
            lines.append(columns(
                kv("output", fmt_int(item.generated)),
                kv("total", fmt_int(item.prompt_tokens + item.generated)),
            ))

    if monitor.last_errors:
        lines.append(section("RECENT ERRORS"))
        lines.extend(row(paint("31", f"• {error}")) for error in monitor.last_errors)

    lines.extend([
        row(""),
        row(paint("2", "Ctrl-C to exit")),
        "╰" + "─" * (box_width - 2) + "╯",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:9000", help="llama-server base URL")
    parser.add_argument("--swap-url", default="http://127.0.0.1:8080", help="llama-swap base URL")
    parser.add_argument("--interval", type=float, default=1.0, help="refresh interval in seconds")
    parser.add_argument("--timeout", type=float, default=1.5, help="HTTP request timeout in seconds")
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    args = parser.parse_args()
    if args.interval <= 0 or args.timeout <= 0:
        parser.error("--interval and --timeout must be positive")

    monitor = Monitor(args.server_url, args.swap_url, args.timeout)
    first = True
    try:
        while True:
            snapshot = monitor.poll()
            if not first and not args.once:
                sys.stdout.write("\033[H\033[2J")
            sys.stdout.write(render(monitor, snapshot, color=not args.no_color) + "\n")
            sys.stdout.flush()
            first = False
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
