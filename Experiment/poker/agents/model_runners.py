"""
Different LLMs can be used
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from huggingface_hub import login

class DummyModelRunner:
    def __init__(self,fixed_response="fold"): self.fixed_response=fixed_response
    def generate(self,prompt): return self.fixed_response

class LlamaLoRAModelRunner:
    """
    Loads a Llama 8B model (4-bit) with optional LoRA adapter.
    Uses HF Transformers and PEFT.
    """

    def __init__(self, base_model: str, adapter_dir: str = None, max_new_tokens: int = 64):
        self.max_new_tokens = max_new_tokens
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Using device: {device}, dtype: {dtype}")

        print(f"[LlamaLoRAModelRunner] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

        print(f"[LlamaLoRAModelRunner] Loading base model ({base_model}) ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            load_in_4bit=True,
            device_map="auto"
        )                   

        if adapter_dir and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            print(f"[LlamaLoRAModelRunner] Detected LoRA adapter → Loading {adapter_dir}")
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        else:
            print("[LlamaLoRAModelRunner] No adapter directory or adapter_config.json not found.")

        self.model.eval()

    def generate(self, prompt: str) -> str:
        #print('prompt',prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"][0]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Slice out only the newly generated tokens
        generated_ids = outputs[0][len(input_ids):]
        final_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        #print('f',final_answer)
        return final_answer


