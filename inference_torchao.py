# !pip install para-attn, mmgp, torchao

from torchao.quantization import quantize_, int8_weight_only
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
import argparse
#from .src.pipeline_pe_clone import FluxPipeline
from src.pipeline_pe_clone import FluxPipeline
import torch
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='FLUX image generation with LoRA')
    parser.add_argument('--model_path', type=str, 
                        default="black-forest-labs/FLUX.1-dev",
                        help='Path to pretrained model')
    parser.add_argument('--image_path', type=str,
                        default="assets/1.png",
                        help='Input image path')
    parser.add_argument('--output_path', type=str,
                        default="output.png",
                        help='Output image path')
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--prompt', type=str,
                        default="add a halo and wings for the cat by sksmagiceffects",
                        help="""Different LoRA effects and their example prompts:
    - sksmagiceffects: "add a halo and wings for the cat by sksmagiceffects"
    - sksmonstercalledlulu: "add a red sksmonstercalledlulu hugging the cat"
    - skspaintingeffects: "add a yellow flower on the cat's head and psychedelic colors and dynamic flows by skspaintingeffects"
    - sksedgeeffect: "add yellow flames to the cat by sksedgeeffect"
    """)
    parser.add_argument('--guidance_scale', type=float, default=3.5)
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Number of inference steps')
    parser.add_argument('--lora_name', type=str,
                        choices=['pretrained', 'sksmagiceffects', 'sksmonstercalledlulu', 
                                'skspaintingeffects', 'sksedgeeffect'],
                        default="sksmagiceffects",
                        help='Name of LoRA weights to use. Use "pretrained" for base model only')
    return parser.parse_args()

def main():
    args = parse_args()
    
    pipeline = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        #cache_dir='/workspace/models'
    ).to('cpu')

    # Load and fuse base LoRA weights
    pipeline.load_lora_weights(
        "nicolaus-huang/PhotoDoodle", weight_name="pretrain.safetensors",#cache_dir='/workspace/models'
        )
    pipeline.fuse_lora()
    pipeline.unload_lora_weights()

    # Load selected LoRA effect only if not using pretrained
    if args.lora_name != 'pretrained':
        pipeline.load_lora_weights(
            "nicolaus-huang/PhotoDoodle", weight_name=f"{args.lora_name}.safetensors",#cache_dir='/workspace/models'
            )

    # Model inference acceleration, reduced from 12s to 8s for 20 steps
    apply_cache_on_pipe(
        pipeline,
        residual_diff_threshold=0.12,  # Use a larger value to make the cache take effect
    )

    # Quantize model to avoid GPU 24GB VRAM overflow
    quantize_(pipeline.transformer, int8_weight_only(), device="cuda")
    quantize_(pipeline.text_encoder, int8_weight_only(), device="cuda")
    quantize_(pipeline.text_encoder_2, int8_weight_only(), device="cuda")

    pipeline.to("cuda")
    
    condition_image = Image.open(args.image_path).resize((args.height, args.width)).convert("RGB")

    result = pipeline(
        prompt=args.prompt,
        condition_image=condition_image,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        max_sequence_length=512
    ).images[0]

    result.save(args.output_path)

if __name__ == "__main__":
    main()
