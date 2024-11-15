from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset
import torch
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

def load_model_and_tokenizer(model_name: str):
    """Load the base model and tokenizer"""
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False
    )
    
    # Load model and tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def main():
    model_name = "mistralai/Mistral-7B-v0.1"
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Load your dad's conversation dataset
    dataset = load_dataset('json', data_files='data/training_data.json')
    
    # Function to tokenize the texts
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Training arguments optimized for Mistral
    training_args = TrainingArguments(
        output_dir="training/dad-ai-model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-5,
        fp16=True,
        save_steps=50,
        logging_steps=10,
        save_total_limit=2,
        warmup_steps=50,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_steps=300,
        weight_decay=0.05,
        remove_unused_columns=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )
    
    # Start training
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model("training/dad-ai-model-final")

if __name__ == "__main__":
    main()