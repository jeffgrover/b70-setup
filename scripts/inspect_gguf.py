#!/usr/bin/env python3
"""Inspect GGUF metadata and tensor names without loading weights or NumPy."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct


SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
           7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def inspect(path: Path) -> dict:
    file_bytes = path.stat().st_size
    with path.open('rb') as stream:
        def advance(length):
            if length > file_bytes - stream.tell():
                raise ValueError('Truncated GGUF field')
            stream.seek(length, 1)

        def unpack(kind):
            size = struct.calcsize('<' + kind)
            value = stream.read(size)
            if len(value) != size:
                raise ValueError('Truncated GGUF header')
            return struct.unpack('<' + kind, value)[0]

        def string(keep=True):
            length = unpack('Q')
            if length > file_bytes - stream.tell():
                raise ValueError('Truncated GGUF string')
            if not keep:
                advance(length)
                return None
            value = stream.read(length)
            if len(value) != length:
                raise ValueError('Truncated GGUF string')
            return value.decode('utf-8')

        def value(kind, keep=True):
            if kind in SCALARS:
                result = unpack(SCALARS[kind])
                return result if keep else None
            if kind == 8:
                return string(keep)
            if kind == 9:
                element, count = unpack('I'), unpack('Q')
                if not keep and element in SCALARS:
                    advance(count * struct.calcsize('<' + SCALARS[element]))
                    return None
                items = []
                for _ in range(count):
                    item = value(element, keep)
                    if keep:
                        items.append(item)
                return items if keep else None
            raise ValueError(f'Unknown GGUF value type {kind}')

        if stream.read(4) != b'GGUF':
            raise ValueError('Not a little-endian GGUF file')
        version = unpack('I')
        if version not in (2, 3):
            raise ValueError(f'Unsupported GGUF version {version}')
        tensor_count, field_count = unpack('Q'), unpack('Q')
        metadata = {}
        for _ in range(field_count):
            name, kind = string(), unpack('I')
            keep = not name.startswith(('tokenizer.ggml.tokens', 'tokenizer.ggml.scores',
                                        'tokenizer.ggml.token_type', 'tokenizer.ggml.merges'))
            result = value(kind, keep)
            if keep:
                metadata[name] = result
        tensors, types = [], Counter()
        for _ in range(tensor_count):
            name = string()
            shape = [unpack('Q') for _ in range(unpack('I'))]
            kind, offset = unpack('I'), unpack('Q')
            types[kind] += 1
            if 'nextn' in name or 'mtp' in name:
                tensors.append(dict(name=name, shape=shape, type_id=kind, offset=offset))
        template = metadata.pop('tokenizer.chat_template', '')
        return dict(path=str(path), bytes=path.stat().st_size, version=version,
                    tensor_count=tensor_count, tensor_types=dict(types),
                    metadata=metadata, nextn_tensors=tensors,
                    chat_template=template,
                    chat_template_sha256=hashlib.sha256(template.encode()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps([inspect(path) for path in args.paths], indent=2))
