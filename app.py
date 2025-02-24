import spaces
import gradio as gr
import torch
from PIL import Image
from src.pipeline_pe_clone import FluxPipeline
import os
import huggingface_hub
huggingface_hub.login(os.getenv('HF_TOKEN_FLUX2'))
# Load default image from assets as an example
default_image = Image.open("assets/1.png")
pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to('cuda')

pipeline.load_lora_weights("nicolaus-huang/PhotoDoodle", weight_name="pretrain.safetensors")
pipeline.fuse_lora()
pipeline.unload_lora_weights()

@spaces.GPU
def generate_image(image, prompt, guidance_scale, num_steps, lora_name):
    # Load the model
    
    # Load and fuse base LoRA weights
    
    # Load selected LoRA effect if not using the pretrained base model
    pipeline.load_lora_weights("nicolaus-huang/PhotoDoodle", weight_name=f"{lora_name}.safetensors")
    pipeline.fuse_lora()
    
    height=768
    width=512

    
    # Prepare the input image
    condition_image = image.resize((height, width)).convert("RGB")
    
    # Generate the output image
    result = pipeline(
        prompt=prompt,
        condition_image=condition_image,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        max_sequence_length=512
    ).images[0]

    final_image  =  result.resize(image.size)
    
    return final_image

# Define examples to be shown within the Gradio interface
examples = [
    # Each example is a list corresponding to the inputs:
    # [Input Image, Prompt, Guidance Scale, Number of Steps, LoRA Name]
    ["assets/1.png", "add a halo and wings for the cat by sksmagiceffects", 3.5, 20, "sksmagiceffects"]
]

# Create Gradio interface with sliders for numeric inputs
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(label="Input Image", type="pil", value=default_image),
        # gr.Slider(label="Height", value=768, minimum=256, maximum=1024, step=64),
        # gr.Slider(label="Width", value=512, minimum=256, maximum=1024, step=64),
        gr.Textbox(label="Prompt", value="add a halo and wings for the cat by sksmagiceffects"),
        gr.Slider(label="Guidance Scale", value=3.5, minimum=1.0, maximum=10.0, step=0.1),
        gr.Slider(label="Number of Steps", value=20, minimum=1, maximum=100, step=1),
        gr.Dropdown(
            label="LoRA Name", 
            choices=["pretrained", "sksmagiceffects", "sksmonstercalledlulu", 
                     "skspaintingeffects", "sksedgeeffect", "skscatooneffect"],
            value="sksmagiceffects"
        )
    ],
    outputs=gr.Image(label="Output Image", type="pil"),
    title="PhotoDoodle-Image-Edit with LoRA",
    examples=examples
)

if __name__ == "__main__":
    iface.launch()
