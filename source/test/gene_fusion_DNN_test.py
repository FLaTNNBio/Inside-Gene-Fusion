import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import logging

class SimpleDNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleDNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def compute_metrics(preds, labels):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def test_model(model, test_dataset, batch_size, num_classes):
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)

    metrics = compute_metrics(all_preds, all_labels)
    logging.info(f"Test Metrics: {metrics}")
    
    # Print the results of predictions and labels
    print("Predictions vs Actual Labels:")
    for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
        print(f"Sample {i + 1}: Predicted = {pred}, Actual = {label}")

    # Return metrics and prediction details for further use if needed
    return all_preds, all_labels, all_probs, metrics

def main():
    logging.basicConfig(
        filename="./logs_test.log",
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Load new data for testing
    new_data = torch.load('new_embeddings_with_labels.pt')  # Adjust the file path as needed
    embeddings = new_data['embeddings']
    labels = new_data['labels']

    # Load the trained model
    input_size = embeddings.shape[1]   # Adjust this to the correct input size
    hidden_size = 128
    output_size = len(torch.unique(labels))    # Adjust this depending on your number of classes
    model = SimpleDNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load("./final_model_DNN.pth"))
    model.eval()

    embeddings = torch.tensor(embeddings)
    labels = torch.tensor(labels)

    test_dataset = TensorDataset(embeddings, labels)

    batch_size = 32
    num_classes = 2  # Change depending on your task (binary or multi-class)

    # Test the model
    logging.info("Start Testing...")
    preds, labels, probs, metrics = test_model(model, test_dataset, batch_size, num_classes)

    # Print final metrics
    print("\nFinal Test Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    logging.info("Testing completed.")

if __name__ == "__main__":
    main()
