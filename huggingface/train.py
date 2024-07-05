# Import necessary libraries
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# Load dataset
dataset = load_dataset("yelp_review_full")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for training.")

# Load pre-trained tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=5)

# Preprocess the data


def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)


encoded_dataset = dataset.map(preprocess_function, batched=True)

# Split the dataset into training and evaluation sets
train_dataset = encoded_dataset["train"].shuffle(seed=42).select(
    range(10000))  # using smaller portion for quick training
eval_dataset = encoded_dataset["test"].shuffle(seed=42).select(range(1000))

# Define compute metrics function for evaluation


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted')
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",        # Align save strategy with evaluation strategy
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    # Load the best model at the end based on the metric below
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)
print("Training on:", trainer.args.device)

# Train and evaluate the model
trainer.train()

# Save the model
model_path = "./sentiment_analysis_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
