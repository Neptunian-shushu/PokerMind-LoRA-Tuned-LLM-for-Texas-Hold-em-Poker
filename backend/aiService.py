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
import re

app = FastAPI(title="PokerMind API", description="LoRA-Tuned LLM for Poker Decision Prediction", version="1.0")

from fastapi.middleware.cors import CORSMiddleware

# CORS
origins = [
    "http://localhost:5173",  
    "http://localhost:5174", 
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          
    allow_credentials=True,
    allow_methods=["*"],           
    allow_headers=["*"],            
)

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

def parse_action_from_text(text: str) -> dict:
    """
    Parse generated text into structured action JSON:
    {
      "action": "FOLD|CHECK|CALL|BET|RAISE",
      "raiseAmount": <number if action is BET or RAISE, 0 otherwise>
    }
    
    Expected model output formats:
    - "bet 18" -> {"action": "BET", "raiseAmount": 18.0}
    - "raise 10" -> {"action": "RAISE", "raiseAmount": 10.0}
    - "call" -> {"action": "CALL", "raiseAmount": 0.0}
    - "fold" -> {"action": "FOLD", "raiseAmount": 0.0}
    - "check" -> {"action": "CHECK", "raiseAmount": 0.0}
    """
    text_lower = text.lower().strip()
    
    # Parse action and amount
    action = None
    raise_amount = 0.0
    
    # Split text into tokens
    tokens = text_lower.split()
    
    if not tokens:
        return {"action": "FOLD", "raiseAmount": 0.0}
    
    first_word = tokens[0]
    
    # Detect action type
    if first_word == "fold":
        action = "FOLD"
    elif first_word == "check":
        action = "CHECK"
    elif first_word == "call":
        action = "CALL"
    elif first_word == "bet":
        action = "BET"
        # Extract the number after bet
        if len(tokens) > 1:
            try:
                raise_amount = float(tokens[1])
            except ValueError:
                # If second token is not a number, look for any number in text
                numbers = re.findall(r'\d+\.?\d*', text)
                if numbers:
                    raise_amount = float(numbers[0])
    elif first_word == "raise":
        action = "RAISE"
        # Extract the number after raise
        if len(tokens) > 1:
            try:
                raise_amount = float(tokens[1])
            except ValueError:
                # If second token is not a number, look for any number in text
                numbers = re.findall(r'\d+\.?\d*', text)
                if numbers:
                    raise_amount = float(numbers[0])
    else:
        # Fallback: search for keywords anywhere in text
        if "fold" in text_lower:
            action = "FOLD"
        elif "check" in text_lower:
            action = "CHECK"
        elif "call" in text_lower:
            action = "CALL"
        elif "bet" in text_lower:
            action = "BET"
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                raise_amount = float(numbers[0])
        elif "raise" in text_lower:
            action = "RAISE"
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                raise_amount = float(numbers[0])
        else:
            # Default to fold if unclear
            action = "FOLD"
    
    return {
        "action": action,
        "raiseAmount": raise_amount
    }

@app.post("/predict")
def predict_action(request: PokerScenario):
    """POST endpoint to predict poker action."""
    try:
        generated_text = predict_poker_action(model, tokenizer, request.instruction)
        parsed_result = parse_action_from_text(generated_text)
        return parsed_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
