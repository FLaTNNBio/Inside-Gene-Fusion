import torch
import numpy as np
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score

# GCN Model definition
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.5):
        super(GCN, self).__init__()
        # Define the GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  # Second GCN layer

        # Dropout layer
        self.dropout = torch.nn.Dropout(p=dropout)  # Dropout with a probability of p

        # Linear layer to classify the entire graph
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # Output dim = 1 for binary classification

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply the first GCN layer followed by ReLU and dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Apply the second GCN layer followed by ReLU and dropout
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Perform global mean pooling over all nodes in the graph
        x = global_mean_pool(x, batch)

        # Apply the final linear layer and return the raw logits
        x = self.fc(x)
        return x

# Function to load a trained model and test it on new data
def test_model(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data).squeeze()

            # Apply sigmoid for binary classification, classify as 1 if prob >= 0.5
            preds = torch.sigmoid(out) >= 0.5
            labels = data.y.view(-1).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    return np.array(all_preds), np.array(all_labels), accuracy

# Function to compute ROC AUC and plot the ROC curve
def compute_roc_auc(preds, labels):
    if len(np.unique(labels)) == 1:
        print("Only one class present in y_true. ROC AUC score is not defined.")
        return None

    roc_auc = roc_auc_score(labels, preds)
    fpr, tpr, _ = roc_curve(labels, preds)
    
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc

# Main function to load model and test it
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the saved GCN model
    model_path = './gcn_model.pth'  # Path to your saved GCN model
    input_dim = 768  # BERT embedding size
    hidden_dim = 128  # Size of hidden layer in GCN
    output_dim = 1    # For binary classification, we need 1 output

    model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Load the graph data
    test_data_path = 'graph_data_1.pth'  # Path to your test graph data
    test_data_list = torch.load(test_data_path)
    
    # Create a DataLoader for the test dataset
    test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)

    # Test the model on new data
    preds, labels, accuracy = test_model(test_loader, model, device)

    # Compute ROC AUC and other metrics
    roc_auc = compute_roc_auc(preds, labels)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    # Print the performance metrics
    print(f"Test ROC AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()
