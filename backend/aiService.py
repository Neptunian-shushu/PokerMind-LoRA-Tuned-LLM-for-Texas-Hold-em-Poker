#!/usr/bin/env python3
"""
FastAPI service for PokerMind model inference
"""

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import uvicorn
import os

app = FastAPI(title="PokerMind API", description="LoRA-Tuned LLM for Poker Decision Prediction", version="1.0")

print("Loading model and tokenizer...")

os.environ["HF_HOME"] = "../../hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "../../hf_cache/transformers"

model_path = "../poker-lora-model/Meta-Llama-3-8B/"
base_model_name = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token=True,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, model_path)
model.eval()

print("PokerMind model loaded successfully.")

class PokerScenario(BaseModel):
    instruction: str

def predict_poker_action(model, tokenizer, instruction):
    """Generate poker action text from input instruction."""
    prompt = instruction
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=inputs['input_ids'].shape[1] + 50,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = response[len(prompt):].strip()
    return generated_text

@app.post("/predict")
def predict_action(request: PokerScenario):
    """POST endpoint to predict poker action."""
    try:
        result = predict_poker_action(model, tokenizer, request.instruction)
        return {"instruction": request.instruction, "predicted_action": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
