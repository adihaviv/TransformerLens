#
# Contains various functions used across analysis scripts (such as model loading, dataset loading, etc)
#

import logging
import sys
from typing import Union
from typing import Optional


from component import Component

from general_utils import (
    balanced_answers_train_test_split,
    get_gpu_count,
    get_image_size_for_model,
    setup_random_counterfactual_prompts,
)



import torch
import transformer_lens as lens
from transformers import (
    Qwen2VLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    MllamaForConditionalGeneration,
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from transformer_lens.HookedVLTransformer import HookedVLTransformer
# or, if the file is in the same folder:
from HookedVLTransformer import HookedVLTransformer


def load_model(
    model_name: str,
    model_path: str,
    device: Union[str, torch.device],
    cache_dir: Optional[str] = None,
    use_tlens_wrapper: bool = True,
    extra_hooks: bool = True,
    torch_dtype: torch.dtype = torch.float32,
):
    if "llama3.2" in model_name.lower():
        inner_model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
        )
        processor = AutoProcessor.from_pretrained(model_path, device=device)
        if use_tlens_wrapper:
            model = HookedVLTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                processor=processor,
                fold_ln=True,
                center_unembed=True,
                center_writing_weights=True,
                fold_value_biases=True,
                n_devices=get_gpu_count(),
                device=device,
            )
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
            model.eval()
        else:
            model = inner_model
        return model, processor

    elif "qwen" in model_name.lower():
        inner_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
        )
        processor = AutoProcessor.from_pretrained(model_path)
        if use_tlens_wrapper:
            inner_model.vision_model = inner_model.visual
            model = HookedVLTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                processor=processor,
                fold_ln=True,
                center_unembed=True,
                center_writing_weights=True,  # False,
                fold_value_biases=True,
                n_devices=get_gpu_count(),
                device=device,
            )
            model.cfg.default_prepend_bos = False  # To match HF Qwen model forward pass
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
            model.eval()
        else:
            model = inner_model
        model.model_name = model_name
        return model, processor

    elif "pixtral" in model_name.lower() or "llava" in model_name.lower():
        inner_model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            #device_map="cpu",
            cache_dir=cache_dir
        )
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        if use_tlens_wrapper:
            inner_model.vision_model = inner_model.vision_tower
            model = HookedVLTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                processor=processor,
                fold_ln=True,
                center_unembed=True,
                center_writing_weights=True,
                fold_value_biases=True,
                n_devices=get_gpu_count(),
                device=device,
            )
            model.cfg.default_prepend_bos = (
                False if "pixtral" in model_name.lower() else True
            )
            # model.cfg.default_prepend_bos = False
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
            model.eval()
        else:
            model = inner_model
        model.model_name = model_name
        return model, processor

    elif "gemma-3" in model_name.lower():
        inner_model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
        )
        processor = AutoProcessor.from_pretrained(model_path)
        # processor.chat_template = processor.chat_template.replace("{{ bos_token }}", "")
        if use_tlens_wrapper:
            inner_model.vision_model = inner_model.vision_tower
            model = HookedVLTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                processor=processor,
                fold_ln=False,  # ,
                center_unembed=True,  # True,
                center_writing_weights=True,  # True,
                fold_value_biases=False,  # True,
                n_devices=get_gpu_count(),
                device=device,
            )
            model.cfg.default_prepend_bos = False
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
            model.eval()
        else:
            model = inner_model
        model.model_name = model_name
        return model, processor

    else:
        print("WARNING: Using model not officially supported in load_model")
        inner_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if use_tlens_wrapper:
            model = HookedTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                tokenizer=tokenizer,
                fold_ln=True,
                center_unembed=True,
                center_writing_weights=True,
                fold_value_biases=True,
                n_devices=get_gpu_count(),
                device=device,
            )
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
        else:
            model = inner_model
        model.model_name = model_name
        return model, tokenizer

