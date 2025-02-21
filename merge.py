
from src.pipeline_pe_clone import FluxPipeline
import torch
from PIL import Image
pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"
pipeline = FluxPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=torch.bfloat16,
)
pipeline.load_lora_weights("outputs/doodle_pretrain_4508000/pytorch_lora_weights.safetensors")
pipeline.fuse_lora()
pipeline.unload_lora_weights()
pipeline.save_pretrained("edit_pretrain")