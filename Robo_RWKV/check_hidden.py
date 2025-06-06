import os
os.environ["RWKV_JIT_ON"] = "1" # Enable RWKV JIT for potential speedup

import json
from PIL import Image
import numpy as np
import math
import argparse
import torch
from pathlib import Path
try:
    from src.rwkv_tokenizer import TRIE_TOKENIZER 
except ImportError:
    raise ImportError("Please ensure TRIE_TOKENIZER is available from src.rwkv_tokenizer or adjust the import path.")

from src.dataset import DEFAULT_IMAGE_TOKEN, DEFAULT_STOP_TOKEN, STOP_TOKEN_INDEX
from src.dataset import process_image_tokens_in_conversations, preprocess
from src.utils import Conversation, gpt4v_crop 

from transformers import CLIPImageProcessor

def get_single_image_tensor_clip(image_path: str, image_processor: CLIPImageProcessor, detail: str = 'low'):
    """
    Loads a single image, processes it using CLIPImageProcessor.
    Handles 'low' and 'high' detail similar to the eval script.
    
    Args:
        image_path (str): Path to the image file.
        image_processor (CLIPImageProcessor): Initialized CLIP image processor.
        detail (str): 'low' or 'high'. 'high' uses gpt4v_crop for more image patches.
        
    Returns:
        torch.Tensor or None: Processed image tensor or None if an error occurs.
    """
    try:
        image = Image.open(image_path).convert("RGB") # Open and convert image to RGB
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    if detail == 'high':
        # For high detail, process the original image and additional crops from gpt4v_crop
        try:
            images_to_process = [image] + gpt4v_crop(image) # Create a list of images (original + crops)
            pixel_values_list = []
            for img_idx, img in enumerate(images_to_process):
                # Process each image/crop using the CLIP image processor
                processed = image_processor(images=img, return_tensors='pt')
                pixel_values_list.append(processed['pixel_values'])
            # Concatenate the tensors from all images/crops along the batch dimension (dim=0)
            image_tensor = torch.cat(pixel_values_list, dim=0) 
        except Exception as e:
            print(f"Error processing high detail image {image_path}: {e}")
            # Fallback to low detail processing if high detail fails
            try:
                print("Falling back to low detail processing for high detail error.")
                image_tensor = image_processor(images=image, return_tensors='pt')['pixel_values']
            except Exception as e_low:
                print(f"Error processing low detail image after high detail fallback: {e_low}")
                return None
    else: # low detail
        try:
            # Process the single image with the CLIP image processor
            image_tensor = image_processor(images=image, return_tensors='pt')['pixel_values']
        except Exception as e:
            print(f"Error processing low detail image {image_path}: {e}")
            return None
            
    return image_tensor


def interactive_demo(args):

    from src.hidden_model import VisualRWKV  

    print("Initializing model for global condition extraction...")
    model_path = Path(args.model_path)

    model = VisualRWKV(args)
    
    try:
        print(f"Loading model weights from... {model_path}") 
        msg = model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        print("Message from loading model weights: ", msg) 
        if hasattr(msg, 'missing_keys') and msg.missing_keys:
             print(f"Warning: Missing keys during model loading: {msg.missing_keys}") 
        if hasattr(msg, 'unexpected_keys') and msg.unexpected_keys:
             print(f"Warning: Unexpected keys during model loading: {msg.unexpected_keys}") 
    except Exception as e:
        print(f"Error loading model weights: {e}") 
        return

    model = model.bfloat16().to(args.device)
    model.eval() 

    try:
        tokenizer = TRIE_TOKENIZER(args.tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}. Please ensure '{args.tokenizer_path}' exists.") 
        return
    try:
        print(f"Loading CLIPImageProcessor from: {args.vision_tower_name}") 
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower_name)
    except Exception as e:
        print(f"Error loading CLIPImageProcessor from '{args.vision_tower_name}': {e}") 
        return

    print("Model and tokenizer initialized for feature extraction.")
    print("Interactive demo started. Type 'quit' to exit.")

    while True:
        try:
            image_path_input = input("Enter image path... > ").strip() 
            if image_path_input.lower() == 'quit':
                break

            user_prompt_input = input(f"Please enter your question/instruction: > ").strip() 
            if user_prompt_input.lower() == 'quit':
                break

            if not image_path_input or not Path(image_path_input).is_file():
                print(f"Image path '{image_path_input}' is invalid or the file does not exist. Please try again.") 
                continue
            if not user_prompt_input:
                print("Prompt cannot be empty. Please try again.") 
                continue
            
            print("Processing image and text to extract global condition...")
            image_tensor = get_single_image_tensor_clip(image_path_input, image_processor, args.detail)
            if image_tensor is None: 
                continue

            image_tensor = image_tensor.unsqueeze(0).bfloat16().to(args.device)

            input_text = DEFAULT_IMAGE_TOKEN + '' + user_prompt_input

            conv = Conversation(id="extraction_session", roles=["human"], conversations=[])
            conv.append_message(conv.roles[0], input_text)
            
            conversations_processed = process_image_tokens_in_conversations(
                conv.conversations, image_position=args.image_position
            )
            
            data_dict = preprocess(
                conversations_processed,
                tokenizer,
                has_image=True,
                ctx_len=args.ctx_len,
                pad_token_id=0,
                do_pad_to_max_length=False
            )

            input_ids = data_dict['input_ids'].unsqueeze(0).to(args.device)
            labels = data_dict['labels'].unsqueeze(0).to(args.device)
            
            samples = {
                "images": image_tensor,
                "input_ids": input_ids,
                "labels": labels
            }

            print("Calling model.get_global_condition...")
            with torch.inference_mode(): 
                global_condition = model.get_global_condition(samples)

            print("--- EXTRACTION RESULT ---")
            print(f"Shape of global_condition: {global_condition.shape}")
            print(f"Dtype of global_condition: {global_condition.dtype}")
            print(f"Device of global_condition: {global_condition.device}")
            
            print(f"  - Min value: {global_condition.min().item():.4f}")
            print(f"  - Max value: {global_condition.max().item():.4f}")
            print(f"  - Mean value: {global_condition.mean().item():.4f}")
            print(f"  - First 5 values: {global_condition.squeeze()[:5].cpu().numpy()}")
            print("-------------------------")
            
            expected_shape = (1, args.n_embd) # Batch size is 1
            if global_condition.shape != expected_shape:
                 print(f"[WARNING] Shape is {global_condition.shape}, but expected {expected_shape}. Check your implementation.")
            else:
                 print("[SUCCESS] The shape of the extracted global condition is correct.")

            torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("\nExiting demo...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

    print("Interactive demo for global condition extraction finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Demo to Extract Global Condition from VisualRWKV Model")
    
    # --- RWKV specific arguments ---
    parser.add_argument("--load_model", default="", type=str)
    parser.add_argument("--vocab_size", default=65536, type=int)
    parser.add_argument("--ctx_len", default=256, type=int)
    parser.add_argument("--n_layer", default=24, type=int)
    parser.add_argument("--n_embd", default=2048, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)
    parser.add_argument("--head_size_a", default=64, type=int)
    parser.add_argument("--head_size_divisor", default=8, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)
    
    # --- VisualRWKV (CLIP version) specific arguments ---
    parser.add_argument("--vision_tower_name", default="/root/zhihuieye/link2data/gyc_SV/VLA/weights/clip-vit-large-patch14-336/", type=str)
    parser.add_argument("--grid_size", type=int, default=-1)
    parser.add_argument("--detail", type=str, default="low", choices=['low', 'high'])
    parser.add_argument("--image_position", default='first', type=str, choices=['first', 'last', 'middle'])
    parser.add_argument("--tokenizer_path", type=str, default="src/rwkv_vocab_v20230424.txt")

    # --- Demo specific arguments ---
    parser.add_argument("--model_path", type=str, default="/root/zhihuieye/link2data/gyc_SV/VLA/weights/VisualRWKV-v060-1B6-v1.0-20240612.pth")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    

    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_STOP_TOKEN = "<|endoftext|>"
    STOP_TOKEN_INDEX = 0 

    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len) 
    
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) 
        print(f"Note: --dim_ffn not set, defaulted to {args.dim_ffn} (3.5 * n_embd, rounded).")

    interactive_demo(args)
