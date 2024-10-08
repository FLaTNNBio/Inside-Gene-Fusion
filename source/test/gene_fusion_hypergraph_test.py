import torch
import pandas as pd
import networkx as nx
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, global_mean_pool
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from transformers import BertTokenizer,BertForSequenceClassification
import ast

class HypergraphNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super(HypergraphNet, self).__init__()

        # Hypergraph convolutional layers using edge_index
        self.hypergraph_conv1 = HypergraphConv(in_channels, hidden_channels)
        self.hypergraph_conv2 = HypergraphConv(hidden_channels, hidden_channels)

        # Linear (fully connected) layer for final classification
        self.fc = nn.Linear(hidden_channels, out_channels)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        # Hypergraph convolution 1
        x = self.hypergraph_conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Hypergraph convolution 2
        x = self.hypergraph_conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Apply global mean pooling to get graph-level representation
        x = global_mean_pool(x, batch)

        # Fully connected layer for output
        out = self.fc(x).view(-1)  # Output a single logit per graph (binary classification)
        return out


def evaluate_and_plot_roc(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []  # For ROC AUC

    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            y = batch.y.to(device).float()
            batch_idx = batch.batch.to(device)

            out = model(x, edge_index, batch_idx).view(-1)  # Get logits
            probs = torch.sigmoid(out)  # Apply sigmoid for probabilities
            preds = probs >= 0.5  # Binarize predictions

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return accuracy, precision, recall, f1, roc_auc


def load_test_data(test_data_path):
    # Load the test dataset
    dataset_with_bert = torch.load(test_data_path)
    test_loader = DataLoader(dataset_with_bert, batch_size=32, shuffle=False)
    return test_loader


def test_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the saved model
    model_save_path = './hypergraph_model.pth'
    input_dim = 128  # Set according to your input feature size
    hidden_dim = 128  # Set hidden dimension
    output_dim = 1  # Binary classification

    model = HypergraphNet(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim)
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()

    # Load the test dataset
    test_data_path = './dataset_with_bert_hyper.pt'
    test_loader = load_test_data(test_data_path)

    # Evaluate model and plot ROC curve
    test_acc, test_prec, test_recall, test_f1, test_roc_auc = evaluate_and_plot_roc(model, test_loader, device)

    # Log the results
    with open('hypergraph_test_metrics.txt', 'a') as f:
        f.write("\n--- Test Performance Metrics ---\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Precision: {test_prec:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write(f"Test F1-Score: {test_f1:.4f}\n")
        f.write(f"Test ROC AUC: {test_roc_auc:.4f}\n")
        f.write(f"-----------------------------------\n")

    # Print results to console
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test ROC AUC: {test_roc_auc:.4f}")


if __name__ == "__main__":
    test_model()
