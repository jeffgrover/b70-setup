import os
import torch
import openvino as ov
import nncf
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM

# Target verified directory
model_path = "/home/jeff/Code/intel/qwen-27b-raw"
output_path = "/home/jeff/Code/intel/qwen3.6-ov-native"

print(f"Step 1: Loading model with Qwen2 architecture fallback...")

os.chdir(model_path)

# Load the config and manually override the model type if Transformers is being stubborn
config = AutoConfig.from_pretrained("./", trust_remote_code=True)
if config.model_type == "qwen3_5":
    print("Found 'qwen3_5' type. Re-mapping to 'qwen2' for compatibility...")
    config.model_type = "qwen2"

tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)

print("Step 2: Loading weights into system RAM (Utilizing that 64GB Swap)...")
# We load specifically as a Qwen2 model to avoid the AutoModel lookup failure
model = Qwen2ForCausalLM.from_pretrained(
    "./", 
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cpu",
    local_files_only=True
)

print("Step 3: Converting to OpenVINO IR and applying INT4 Compression...")
# This step maps the weights to the Battlemage XMX engines
ov_model = ov.convert_model(model)
quantized_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT4_ASYM)

print(f"Step 4: Saving native IR to {output_path}...")
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

ov.save_model(quantized_model, os.path.join(output_path, "openvino_model.xml"))
tokenizer.save_pretrained(output_path)

print("\nConversion successful. The Battlemage B70 is ready to execute.")