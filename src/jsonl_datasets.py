import torch
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
import random
import os
import numpy as np

Image.MAX_IMAGE_PIXELS = None

def make_train_dataset(args, tokenizer, accelerator=None):
    if args.train_data_dir is not None:
        print("load_data")
        dataset = load_dataset('json', data_files=args.train_data_dir)
        base_path = os.path.dirname(os.path.abspath(args.train_data_dir))

    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.caption_column is None:
        caption_column = column_names[0]
        print(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
    if args.source_column is None:
        source_column = column_names[1]
        print(f"source column defaulting to {source_column}")
    else:
        source_column = args.source_column
        if source_column not in column_names:
            raise ValueError(
                f"`--source_column` value '{args.source_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
    if args.target_column is None:
        target_column = column_names[1]
        print(f"target column defaulting to {target_column}")
    else:
        target_column = args.target_column
        if target_column not in column_names:
            raise ValueError(
                f"`--target_column` value '{args.target_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
            
    h = args.height
    w = args.width
    train_transforms = transforms.Compose(
        [
            transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    tokenizer_clip = tokenizer[0]
    tokenizer_t5 = tokenizer[1]

    def tokenize_prompt_clip_t5(examples):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)                    
            elif isinstance(caption, list):
                captions.append(random.choice(caption))
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        text_inputs = tokenizer_clip(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_1 = text_inputs.input_ids

        text_inputs = tokenizer_t5(
            captions,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs.input_ids
        return text_input_ids_1, text_input_ids_2

    def preprocess_train(examples):
        _examples = {}        

        source_images = [Image.open(os.path.join(base_path, image)).convert("RGB") 
                        for image in examples[source_column]]
        target_images = [Image.open(os.path.join(base_path, image)).convert("RGB") 
                        for image in examples[target_column]]

        _examples["cond_pixel_values"] = [train_transforms(source) for source in source_images]
        _examples["pixel_values"] = [train_transforms(image) for image in target_images]
        _examples["token_ids_clip"], _examples["token_ids_t5"] = tokenize_prompt_clip_t5(examples)

        return _examples

    if accelerator is not None:
        with accelerator.main_process_first():
            train_dataset = dataset["train"].with_transform(preprocess_train)
    else:
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
    cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
    target_pixel_values = torch.stack([example["pixel_values"] for example in examples])
    target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()
    token_ids_clip = torch.stack([torch.tensor(example["token_ids_clip"]) for example in examples])
    token_ids_t5 = torch.stack([torch.tensor(example["token_ids_t5"]) for example in examples])

    return {
        "cond_pixel_values": cond_pixel_values,
        "pixel_values": target_pixel_values,
        "text_ids_1": token_ids_clip,
        "text_ids_2": token_ids_t5,
    }
