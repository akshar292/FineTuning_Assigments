# Task 4: QLoRA Setup Example

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model
)


# Model
model_name = "distilbert-base-uncased"


# 4-bit quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    quantization_config=quantization_config
)


# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)


# Apply LoRA
model = get_peft_model(
    model,
    lora_config
)


# Display trainable parameters
model.print_trainable_parameters()