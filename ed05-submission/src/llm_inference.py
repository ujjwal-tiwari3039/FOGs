import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
LORA_DIR = "./models/qwen_lora"

class CHiLLGrader:
    def __init__(self, use_lora=True):
        print("Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # Token IDs for the 3 classes (we grab the first token of the word)
        self.class_tokens = {
            'correct': self.tokenizer.encode("correct", add_special_tokens=False)[0],
            'contradictory': self.tokenizer.encode("contradictory", add_special_tokens=False)[0],
            'incorrect': self.tokenizer.encode("incorrect", add_special_tokens=False)[0]
        }
        
        print("Loading 4-bit Base Model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        if use_lora:
            try:
                print("Attaching LoRA Adapters...")
                self.model = PeftModel.from_pretrained(self.model, LORA_DIR)
            except Exception as e:
                print(f"No LoRA adapters found at {LORA_DIR}. Using base model. Error: {e}")
                
        self.model.eval()

    def format_prompt(self, q, r, s):
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

    @torch.no_grad()
    def predict(self, q, refs, s, temperature=1.0):
        # We use the best reference or join them
        best_ref = refs[0] if refs else ""
        prompt = self.format_prompt(q, best_ref, s)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Forward pass to get logits for the next token
        outputs = self.model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        
        # Extract logits for our 3 specific target classes
        target_logits = {
            label: next_token_logits[tok_id].item() 
            for label, tok_id in self.class_tokens.items()
        }
        
        # Apply Softmax with Temperature Scaling (Calibration)
        logits_arr = np.array([target_logits['correct'], target_logits['contradictory'], target_logits['incorrect']])
        scaled_logits = logits_arr / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits)) # Stable softmax
        probs = exp_logits / exp_logits.sum()
        
        prob_dict = {
            'correct': probs[0],
            'contradictory': probs[1],
            'incorrect': probs[2]
        }
        
        pred_label = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[pred_label]
        
        return pred_label, confidence, prob_dict
