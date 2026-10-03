# Task 1: Check Trainable Parameters Before and After LoRA

import torch

from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


# Load pre-trained sentiment classification model
model_name = "distilbert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# Function to count parameters
def count_parameters(model):

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_params, trainable_params


# Parameters before LoRA
total_before, trainable_before = count_parameters(model)

print("Before LoRA")
print("Total Parameters:", total_before)
print("Trainable Parameters:", trainable_before)


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
model = get_peft_model(model, lora_config)


# Parameters after LoRA
total_after, trainable_after = count_parameters(model)

print("\nAfter LoRA")
print("Total Parameters:", total_after)
print("Trainable Parameters:", trainable_after)


# Display percentage
percentage = (
    trainable_after / total_after
) * 100

print(
    f"Trainable Parameters Percentage: "
    f"{percentage:.2f}%"
)