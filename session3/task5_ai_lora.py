# Task 5: Apply LoRA to a Text Classification Model

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from peft import (
    LoraConfig,
    get_peft_model
)


# Load model
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
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


# Display parameters
model.print_trainable_parameters()


# Save LoRA adapter
model.save_pretrained("./my_lora_model")

print("LoRA adapter saved successfully!")