from transformers import BertTokenizer,BertForSequenceClassification
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd

def DNABERT_setting(model_name,nl):
    #model_name = "zhihan1996/DNA_bert_6"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=nl)
    return tokenizer, model


# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Get the index of the highest logit
    accuracy = accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy
    }

def create_dataset(df_fused,tokenizer):
    # Assuming df_fused contains your 6-merized DNA sequences and labels
    sequences = df_fused['kmerized_sequence'].tolist()  # List of DNA sequences (6-mers)
    labels = df_fused['label'].tolist()  # List of labels (0 or 1)

    # Tokenize all sequences at once
    encodings = tokenizer(sequences, padding='max_length', truncation=True, max_length=100, return_tensors='pt')

    # Check the structure of the encodings (contains input_ids and attention_mask)
    print(encodings.keys())  # Should print: dict_keys(['input_ids', 'attention_mask'])

    # Convert the labels to tensor format
    labels = torch.tensor(labels)

    # Create the DNADataset class
    class DNADataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    # Create the dataset using the encodings and labels
    dataset = DNADataset(encodings, labels)
    return dataset

def train_DNABERT(model,tokenizer, dataset):
    # Assuming `dataset` is a PyTorch dataset or a list of graphs created in the previous step
    train_indices, eval_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

    # Create subsets for training and evaluation
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    eval_dataset = torch.utils.data.Subset(dataset, eval_indices)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # Output directory
        num_train_epochs=5,              # Number of training epochs
        per_device_train_batch_size=16,  # Batch size for training
        per_device_eval_batch_size=16,   # Batch size for evaluation
        warmup_steps=500,                # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # Strength of weight decay
        logging_dir='./logs',            # Directory for storing logs
        logging_steps=10,                # Log every 10 steps
        eval_strategy="epoch",     # Evaluate every epoch
        save_steps=500                   # Save every 500 steps
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,                         # The pre-trained model with a classification head
        args=training_args,                  # Training arguments
        train_dataset=train_dataset,         # The dataset for training
        eval_dataset=eval_dataset,           # The dataset for evaluation
        tokenizer=tokenizer,                 # The tokenizer used
        compute_metrics=compute_metrics      # Pass the custom metrics function
    )

    # Train the model
    trainer.train()



def load_DNABERT_model(model_name):

    # Load the fine-tuned model and tokenizer
    #model_name = "./fine_tuned_dna_bert"  # Path to your fine-tuned model

    # Load the fine-tuned model for sequence classification
    load_model = BertForSequenceClassification.from_pretrained(model_name)
    return load_model

def main():
    df_fused = pd.read_csv('./df_fused.csv')
    tokenizer, model = DNABERT_setting("zhihan1996/DNA_bert_6",2)
    dataset = create_dataset(df_fused,tokenizer)
    train_DNABERT(model,tokenizer, dataset)
    model.save_pretrained('./fine_tuned_fusion_dna_bert')

if __name__ == "__main__":
    main()