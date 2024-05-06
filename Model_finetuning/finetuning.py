import torch
import os
import sys
import json
from datetime import datetime
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Choose the base model you want
model_name = "Hugofernandez/Mistral-7B-v0.1-colab-sharded"

# Set device
device = 'cuda'

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# We redefine the pad_token and pad_token_id with out of vocabulary token (unk_token)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id


compute_dtype = getattr(torch, "float16")
print(compute_dtype)
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config, 
    use_flash_attention_2=False,  # Set to True if you're using A100
    device_map={"": 0}  # Manually setting the model to be on GPU 0
)

peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj", "lm_head",]
)

# Cast some modules of the model to fp32 
model = prepare_model_for_kbit_training(model)
# Configure the pad token in the model
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False # Gradient checkpoint

training_arguments = TrainingArguments(
        output_dir="./results", # Directory in which the checkpoint will be saved. 
        evaluation_strategy="epoch", # You can set it to 'steps' to evaluate it every eval_steps 
        optim="paged_adamw_8bit", # Used with QLoRA
        per_device_train_batch_size=4, # Batch size 
        per_device_eval_batch_size=4, # Same but for evaluation 
        gradient_accumulation_steps=1, # Number of lines to accumulate gradient, careful because it changes the size of a "step". Therefore, logging, evaluation, save will be conducted every gradient_accumulation_steps * xxx_step training example
        log_level="debug", # You can set it to  ‘info’, ‘warning’, ‘error’ and ‘critical’ 
        save_steps=500, # Number of steps between checkpoints 
        logging_steps=20, # Number of steps between logging of the loss for monitoring, adapt it to your dataset size
        learning_rate=4e-4, # You can try different values for this hyperparameter
        num_train_epochs=1,
        warmup_steps=100,
        lr_scheduler_type="constant",
)


data=pd.read_csv("data.csv")

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.1)

# Create Hugging Face dataset objects
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

def tokenize_function(examples):
    # Keeping all the text, not truncating to 512 as it's an LLM capable of handling more
    model_inputs = tokenizer(examples['input'], truncation=False)
    with tokenizer.as_target_tokenizer():
        # Output might need to be handled as per your JSON structure and model requirements
        labels = tokenizer(examples['output'], truncation=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, return_tensors="pt")
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    peft_config = peft_config,
    tokenizer=tokenizer,
    data_collator=data_collator,
    dataset_text_field='input'  # Assuming 'input' is the field in your dataset containing the text for training
)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = model.num_parameters()
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
print_trainable_parameters(model)

trainer.evaluate()

trainer.train()


eval_text = "Extract the entities and relationships from the following text:The levels of NLR, PLR, SII, and NLPR were significantly higher in cancer patients than in healthy controls."
# Prepare the input text for evaluation
model_input = tokenizer(eval_text, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    # Assuming the model outputs structured data representing entities and relationships
    # Note: The interpretation of the output will depend on your specific model's format
    output_tokens = model.generate(**model_input, max_new_tokens=256, pad_token_id=tokenizer.pad_token_id)[0]
    extracted_data = tokenizer.decode(output_tokens, skip_special_tokens=True)
    print(extracted_data)
model.train()
