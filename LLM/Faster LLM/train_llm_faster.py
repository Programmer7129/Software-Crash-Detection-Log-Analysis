from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load the preprocessed data
with open('log_entries.txt', 'r') as f:
    log_entries = f.readlines()

# Create a Dataset object
dataset = Dataset.from_dict({'text': log_entries})

# Tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add a padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Add labels to the dataset
def add_labels(examples):
    examples['labels'] = examples['input_ids'].copy()
    return examples

tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)

# Load model and resize token embeddings
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Ensure the model uses CPU
device = torch.device('cpu')
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,  # Use fewer epochs
    per_device_train_batch_size=4,
    save_steps=1_000,  # Save model more frequently
    save_total_limit=2,
    use_cpu=True,  # Ensure no CUDA is used
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
