import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

def format_prompt(q, r, s):
    return f"""<|im_start|>system
You are an expert teacher grading short answers. Output exactly one word: 'correct', 'contradictory', or 'incorrect'.
<|im_end|>
<|im_start|>user
Question: {q}
Reference Answer: {r}
Student Answer: {s}
<|im_end|>
<|im_start|>assistant
"""

def prepare_dataset(records):
    data = {"text": []}
    for rec in records:
        refs = rec['reference_answers']
        best_ref = refs[0] if refs else ""
        prompt = format_prompt(rec['question_text'], best_ref, rec['student_answer'])
        data["text"].append(prompt + rec['label'] + "<|im_end|>")
    return Dataset.from_dict(data)

def train_lora(train_records, output_dir="./models/qwen_lora"):
    print(f"Loading {MODEL_ID} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    dataset = prepare_dataset(train_records)
    
    from trl import SFTConfig
    
    args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1, # 6GB VRAM constraint
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=50, # Demo steps
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        optim="paged_adamw_8bit",
        dataset_text_field="text",
        max_length=512,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=args,
    )
    
    print("Starting LoRA fine-tuning...")
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    print("Training complete!")

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path
    
    # Add project root to sys.path so we can import src modules
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.load_data import load_training_data
    
    print(f"Loading training data for {MODEL_ID} LoRA Fine-tuning...")
    data_dir = Path(__file__).parent.parent / "data" / "semeval-2013-task7" / "semeval-3way"
    
    if not data_dir.exists():
        print("Error: Dataset not found. Please ensure SemEval data is in ed05-submission/data/")
        sys.exit(1)
        
    train_records = load_training_data(data_dir)
    print(f"Found {len(train_records)} training records.")
    
    # Run the LoRA training
    train_lora(train_records)
