#!/usr/bin/env python3
"""
Llama-3-8B LoRA Fine-tuning on PokerBench Dataset
CS6220 Project - PACE ICE Cluster
Author: bshu30
Description: Fine-tune Meta-Llama-3-8B model on poker decision-making using LoRA
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np
import pandas as pd
import json
import os
import warnings
from tqdm.auto import tqdm
import gc
import sys
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    print("="*80)
    print("Llama-3-8B LoRA Fine-tuning on PokerBench Dataset")
    print("CS6220 Project - PACE ICE Cluster")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # ========== 1. Environment Setup ==========
    print("\n[1/9] Checking environment...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using NVIDIA GPU (CUDA): {torch.cuda.get_device_name()}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)
    else:
        print("ERROR: No CUDA device found. This script requires GPU.")
        sys.exit(1)
    
    device = torch.device("cuda")
    
    # ========== 2. Load Dataset ==========
    print("\n[2/9] Loading PokerBench dataset...")
    dataset = load_dataset("RZ412/PokerBench")
    print(f"✓ Dataset loaded - Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    
    # ========== 3. Configuration ==========
    print("\n[3/9] Setting up configuration...")
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    MAX_LENGTH = 512
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    
    # Create balanced training set with both preflop and postflop examples
    POSTFLOP_SAMPLES = 80000  # Postflop examples from beginning (0-79999)
    PREFLOP_SAMPLES = 30000   # Preflop examples from end (last 30000 rows)
    TOTAL_SUBSET = POSTFLOP_SAMPLES + PREFLOP_SAMPLES  # 110,000 total samples
    TRAIN_RATIO = 0.9    # 90% for training, 10% for validation
    
    # Calculate split sizes
    TRAIN_SAMPLES = int(TOTAL_SUBSET * TRAIN_RATIO)  # 99,000 samples
    VAL_SAMPLES = TOTAL_SUBSET - TRAIN_SAMPLES        # 11,000 samples
    
    print(f"✓ Model: {MODEL_NAME}")
    print(f"✓ LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"✓ Dataset subset: {POSTFLOP_SAMPLES} postflop + {PREFLOP_SAMPLES} preflop = {TOTAL_SUBSET} total")
    print(f"✓ Split: {TRAIN_SAMPLES} train ({TRAIN_RATIO*100:.0f}%), {VAL_SAMPLES} val ({(1-TRAIN_RATIO)*100:.0f}%)")
    
    # ========== 4. Load Tokenizer ==========
    print("\n[4/9] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded - Vocab size: {len(tokenizer):,}")
    
    # ========== 5. Load Model ==========
    print("\n[5/9] Loading model with 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded with 4-bit quantization - {total_params:,} parameters")
    
    # ========== 6. Apply LoRA ==========
    print("\n[6/9] Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    print("✓ LoRA applied:")
    model.print_trainable_parameters()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ========== 7. Prepare Dataset ==========
    print("\n[7/9] Preprocessing dataset...")
    
    def format_prompt(instruction, output):
        """Format instruction-output pairs for fine-tuning"""
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    # Select balanced training data: postflop + preflop examples
    postflop_train_data = dataset['train'].select(range(POSTFLOP_SAMPLES))  # First 80,000 (postflop)
    preflop_start_idx = len(dataset['train']) - PREFLOP_SAMPLES  # Last 30,000 (preflop)
    preflop_train_data = dataset['train'].select(range(preflop_start_idx, len(dataset['train'])))
    
    # Combine postflop and preflop data
    train_full = concatenate_datasets([postflop_train_data, preflop_train_data])
    print(f"  Combined training data: {len(postflop_train_data)} postflop + {len(preflop_train_data)} preflop = {len(train_full)} total")
    
    # Shuffle and split into train/val
    train_full = train_full.shuffle(seed=42)
    train_dataset = train_full.select(range(TRAIN_SAMPLES))
    val_dataset = train_full.select(range(TRAIN_SAMPLES, TOTAL_SUBSET))
    
    print(f"  Split into {len(train_dataset)} train and {len(val_dataset)} validation samples")
    
    def preprocess_function(examples):
        """Tokenize examples - padding will be done dynamically by data collator"""
        prompts = [format_prompt(inst, out) for inst, out in zip(examples['instruction'], examples['output'])]
        model_inputs = tokenizer(prompts, max_length=MAX_LENGTH, truncation=True, padding=False)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)
    
    print(f"✓ Tokenization complete - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # ========== 8. Training Setup ==========
    print("\n[8/9] Configuring training...")
    output_dir = "../poker-lora-model/Meta-Llama-3-8B/"
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,               # 3 epoch for training
        per_device_train_batch_size=16,   # 16 samples per GPU per step
        per_device_eval_batch_size=32,    # Larger batch for evaluation (no gradients)
        gradient_accumulation_steps=8,    # Effective batch size = 16 * 8 = 128
        learning_rate=2e-4,
        warmup_steps=100,                 # ~8% of total steps (99k/128 = 773 steps)
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,         # More workers for faster data loading
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=50,                 # Log every 50 steps (~6% of epoch)
        save_strategy="no",               # Disable checkpoint saving
        eval_strategy="steps",            # Evaluate at regular intervals
        eval_steps=200,                   # Evaluate every 200 steps (~4 times per epoch)
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
        max_grad_norm=1.0,
        weight_decay=0.01,
        seed=42,
    )
    
    # Create data collator with dynamic padding for variable-length sequences
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,  # Enable dynamic padding
        pad_to_multiple_of=8,  # Pad to multiples of 8 for efficient GPU computation
        return_tensors="pt"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print(f"✓ Training config ready - Output: {output_dir}")
    
    # ========== 9. Train Model ==========
    print("\n[9/9] Starting training...")
    print("="*80)
    
    training_result = trainer.train()
    
    print("="*80)
    print(f"✓ Training completed - Final loss: {training_result.training_loss:.4f}")
    trainer.save_metrics("train", training_result.metrics)
    
    # ========== 10. Evaluation ==========
    print("\n Evaluating model performance...")
    
    def extract_action_and_amount(output_text):
        """Extract action and amount from model output"""
        tokens = output_text.strip().lower().split()
        if not tokens:
            return None, None
        action = tokens[0]
        amount = None
        if action in ['bet', 'raise', 'call'] and len(tokens) > 1:
            try:
                amount = float(tokens[1])
            except (ValueError, IndexError):
                pass
        return action, amount
    
    def analyze_preflop_postflop(model, tokenizer, preflop_dataset, postflop_dataset):
        """Analyze performance on preflop vs postflop scenarios"""
        model_device = next(model.parameters()).device
        
        # Disable gradient checkpointing for evaluation to enable KV-cache
        model.gradient_checkpointing_disable()
        
        def process_dataset(dataset, stats, scenario_name):
            print(f"  Evaluating {len(dataset)} {scenario_name} samples... ", end='', flush=True)
            for i, example in enumerate(dataset):
                stats['total'] += 1
                prompt = format_prompt(example['instruction'], "")
                
                try:
                    # Tokenize with proper attention mask
                    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model_device)
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
                            use_cache=True  # Enable KV-cache for faster generation
                        )
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_part = response[len(prompt):].strip()
                    
                    expected_action, expected_amount = extract_action_and_amount(example['output'])
                    generated_action, generated_amount = extract_action_and_amount(generated_part)
                    
                    if expected_action is None:
                        continue
                    
                    if expected_action in stats['actions']:
                        stats['actions'][expected_action] += 1
                    
                    if expected_action == generated_action:
                        stats['action_correct'] += 1
                        if expected_action in ['bet', 'raise', 'call']:
                            if expected_amount is not None and generated_amount is not None:
                                if abs(expected_amount - generated_amount) < 0.01:
                                    stats['exact_match_correct'] += 1
                            elif expected_amount is None and generated_amount is None:
                                stats['exact_match_correct'] += 1
                        else:
                            stats['exact_match_correct'] += 1
                except:
                    continue
        
        preflop_stats = {'total': 0, 'action_correct': 0, 'exact_match_correct': 0,
                         'actions': {'fold': 0, 'call': 0, 'bet': 0, 'raise': 0, 'check': 0}}
        postflop_stats = {'total': 0, 'action_correct': 0, 'exact_match_correct': 0,
                          'actions': {'fold': 0, 'call': 0, 'bet': 0, 'raise': 0, 'check': 0}}
        
        process_dataset(preflop_dataset, preflop_stats, "preflop")
        print("Done!")
        
        process_dataset(postflop_dataset, postflop_stats, "postflop")
        print("Done!")
        
        return preflop_stats, postflop_stats
    
    def display_scenario_stats(stats, scenario_name):
        """Display statistics for a poker scenario"""
        if stats['total'] == 0:
            print(f"\n{scenario_name}: No samples found")
            return
        
        aa = (stats['action_correct'] / stats['total']) * 100
        em = (stats['exact_match_correct'] / stats['total']) * 100
        
        print(f"\n{scenario_name} Performance:")
        print(f"  Total Samples: {stats['total']}")
        print(f"  Action Accuracy: {aa:.2f}% ({stats['action_correct']}/{stats['total']})")
        print(f"  Exact Match Accuracy: {em:.2f}% ({stats['exact_match_correct']}/{stats['total']})")
        print(f"  Action Distribution:")
        for action, count in stats['actions'].items():
            if count > 0:
                print(f"    {action.upper()}: {count} ({(count / stats['total']) * 100:.1f}%)")
    
    postflop_test = dataset['test'].select(range(0, 1000))
    preflop_test = dataset['test'].select(range(10000, len(dataset['test'])))
    
    preflop_stats, postflop_stats = analyze_preflop_postflop(model, tokenizer, preflop_test, postflop_test)
    
    display_scenario_stats(preflop_stats, "PREFLOP")
    display_scenario_stats(postflop_stats, "POSTFLOP")
    
    if preflop_stats['total'] > 0 and postflop_stats['total'] > 0:
        preflop_acc = (preflop_stats['action_correct'] / preflop_stats['total']) * 100
        postflop_acc = (postflop_stats['action_correct'] / postflop_stats['total']) * 100
        print(f"\nOverall - Preflop: {preflop_acc:.1f}%, Postflop: {postflop_acc:.1f}%")
    
    # ========== 11. Save Model ==========
    print("\n[FINAL] Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    print(f"✓ Model saved to {output_dir} ({total_size / (1024**2):.2f} MB)")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
