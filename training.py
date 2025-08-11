import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import pandas as pd
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        conversation = item['conversation']
        full_text = ""
        
        for turn in conversation:
            if turn['role'] == 'human':
                full_text += f"<s>[INST] {turn['content']} [/INST]"
            else:
                full_text += f"{turn['content']}</s>"
        
        risk_category = item['risk_calculation']['risk_category']
        full_text += f" Risk Category: {risk_category}"
        
        formatted_data.append({
            "text": full_text.strip()
        })
    
    return Dataset.from_pandas(pd.DataFrame(formatted_data))

def initialize_model_and_tokenizer():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False
    )
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def configure_lora(model):
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
        modules_to_save=None  # Changed from previous version
    )
    
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    
    # Enable input requires_grad
    model.config.use_cache = False
    
    print("Trainable parameters:")
    model.print_trainable_parameters()
    return model

def tokenize_dataset(dataset, tokenizer):
    print("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def setup_training_args():
    return TrainingArguments(
        output_dir="./mistral-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=1,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        optim="paged_adamw_32bit",
        ddp_find_unused_parameters=False
    )

def main():
    try:
        print("\nStarting training process...")
        print("Step 1: Loading dataset")
        dataset = load_dataset("investment_profiles.json")
        
        print("\nStep 2: Initializing model and tokenizer")
        model, tokenizer = initialize_model_and_tokenizer()
        
        print("\nStep 3: Applying LoRA configuration")
        model = configure_lora(model)
        
        print("\nStep 4: Tokenizing dataset")
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)
        
        print("\nStep 5: Setting up training arguments")
        training_args = setup_training_args()
        
        print("\nStep 6: Initializing data collator")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        print("\nStep 7: Setting up trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        print("\nStep 8: Starting training")
        trainer.train()
        
        print("\nStep 9: Saving model")
        trainer.save_model("./final_model")
        tokenizer.save_pretrained("./final_model")
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())
        raise e

if __name__ == "__main__":
    # Print system info
    print("\nSystem Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Run main process
    main()