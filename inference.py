
from src.pipeline_scale import FluxPipeline
import torch
from PIL import Image
pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"
pipeline = FluxPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=torch.bfloat16,
).to('cuda')
pipeline.load_lora_weights("outputs/doodle_pretrain_4508000")
pipeline.fuse_lora()
pipeline.unload_lora_weights()
pipeline.load_lora_weights("outputs/lora2/checkpoint-8000")
height=768
width=512
validation_image = "assets/1.png"
validation_prompt = "add a halo and wings for the cat by sksmagiceffects"
condition_image = Image.open(validation_image).resize((height, width)).convert("RGB")

result = pipeline(prompt=validation_prompt, 
                  condition_image=condition_image,
                  height=height,
                  width=width,
                  guidance_scale=3.5,
                  num_inference_steps=20,
                  max_sequence_length=512).images[0]

result.save("output.png")