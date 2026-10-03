# Task 2: LoRA Fine-Tuning on a Small Tweet Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from peft import (
    LoraConfig,
    get_peft_model
)

from datasets import Dataset


# --------------------------------
# 1. Custom Dataset
# --------------------------------

texts = [
    "I love this product",
    "Amazing experience",
    "This is fantastic",
    "Very happy with the service",
    "Excellent product",
    "I hate this product",
    "Very bad experience",
    "This is terrible",
    "Worst service ever",
    "I am disappointed"
]


labels = [
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0
]


dataset = Dataset.from_dict({
    "text": texts,
    "label": labels
})


# --------------------------------
# 2. Load Tokenizer
# --------------------------------

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


# --------------------------------
# 3. Tokenize Dataset
# --------------------------------

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)


# --------------------------------
# 4. Load Model
# --------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# --------------------------------
# 5. LoRA Configuration
# --------------------------------

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


# --------------------------------
# 6. Training Arguments
# --------------------------------

training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_steps=1,
    save_strategy="no",
    report_to="none"
)


# --------------------------------
# 7. Trainer
# --------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)


# --------------------------------
# 8. Train
# --------------------------------

print("\nStarting LoRA training...")

trainer.train()


# --------------------------------
# 9. Save LoRA Adapter
# --------------------------------

output_directory = "./lora_sentiment_model"

model.save_pretrained(output_directory)

tokenizer.save_pretrained(output_directory)


print("\nLoRA model saved to:")
print(output_directory)