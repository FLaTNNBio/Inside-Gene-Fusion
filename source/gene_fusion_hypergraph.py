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

# Step 2: Helper function to build a graph from De Bruijn edges
def build_de_bruijn_graph(edges):
    G = nx.Graph()  # Create an undirected graph
    # Add edges to the graph
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    return G

# Step 3: Helper function to find cliques and create a hypergraph
def create_hypergraph_from_cliques(edges):
    # Build the graph from De Bruijn edges
    G = build_de_bruijn_graph(edges)

    # Find all cliques in the graph
    cliques = list(nx.find_cliques(G))

    # Create a hypergraph where each clique is a hyperedge
    hypergraph = {f"hyperedge_{i}": set(clique) for i, clique in enumerate(cliques)}

    return hypergraph

# Step 4: Process each row in the dataset
def process_dataset(df):
    hypergraphs = []

    for index, row in df.iterrows():
        kmer_sequence = row[0]  # The k-mer sequence (not used in hypergraph creation here)
        label = row[1]          # The label
        debruijn_edges = eval(row[2])  # Convert string representation of edges to set of tuples

        # Create hypergraph from cliques in the graph built from De Bruijn edges
        hypergraph = create_hypergraph_from_cliques(debruijn_edges)

        # Store the hypergraph (you can also store it alongside the label if needed)
        hypergraphs.append({'index': index, 'hypergraph': hypergraph, 'label': label})

    return hypergraphs


def get_feature_matrix_from_hypergraph(hypergraph, model, tokenizer, device, batch_size=32):
    model.eval()  # Set BERT model to evaluation mode
    features_matrix = []

    # Collect all unique nodes (k-mers) from the hypergraph
    nodes = set()
    for hyperedge in hypergraph.values():
        nodes.update(hyperedge)
    nodes = list(nodes)  # Convert set of nodes to a list

    # Move the BERT model to the correct device
    model = model.to(device)

    # Batch processing to speed up tokenization and inference
    for i in range(0, len(nodes), batch_size):
        batch_nodes = nodes[i:i+batch_size]

        # Tokenize the batch of nodes (k-mers)
        inputs = tokenizer(batch_nodes, return_tensors="pt", max_length=128, padding='max_length', truncation=True)

        # Move inputs to the appropriate device (GPU/CPU)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Extract BERT embeddings without computing gradients
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get the hidden states (last layer)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]

        # Mean pooling: Take the mean of token embeddings across the sequence length (dim=1)
        pooled_embeddings = last_hidden_state.mean(dim=1)

        # Append each node's embedding to the features matrix
        features_matrix.extend(pooled_embeddings.cpu().numpy())  # Move to CPU before converting to NumPy

    # Convert features matrix to a tensor
    return torch.tensor(features_matrix, dtype=torch.float), nodes


def prepare_hypergraph_with_bert(hypergraph_results, bert_model, tokenizer, device):
    data_list = []

    for idx, result in enumerate(hypergraph_results):
        hypergraph = result['hypergraph']
        label = result['label']

        # Get BERT embeddings for the nodes (k-mers)
        features_matrix, nodes = get_feature_matrix_from_hypergraph(hypergraph, bert_model, tokenizer, device)

        # Create node-to-index mapping
        node_index = {node: i for i, node in enumerate(nodes)}

        # Create edge index (connect all pairs within each hyperedge)
        edges = []
        for hyperedge in hypergraph.values():
            hyperedge_nodes = list(hyperedge)
            for i in range(len(hyperedge_nodes)):
                for j in range(i + 1, len(hyperedge_nodes)):
                    edges.append([node_index[hyperedge_nodes[i]], node_index[hyperedge_nodes[j]]])

        # Create edge_index tensor for PyTorch Geometric (of shape [2, num_edges])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

        # Label (binary classification)
        y = torch.tensor([label], dtype=torch.long).to(device)

        # Create PyTorch Geometric Data object for each hypergraph
        data = Data(x=features_matrix, edge_index=edge_index, y=y)
        print(data)
        data_list.append(data)

    return data_list

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
        x = self.hypergraph_conv1(x, edge_index)  # Use edge_index as the connectivity index
        x = F.relu(x)
        x = self.dropout(x)

        # Hypergraph convolution 2
        x = self.hypergraph_conv2(x, edge_index)  # Use edge_index again
        x = F.relu(x)
        x = self.dropout(x)

        # Apply global mean pooling to get graph-level representation
        x = global_mean_pool(x, batch)  # Pooling over nodes to get a single embedding per graph

        # Fully connected layer for output
        out = self.fc(x).view(-1)  # Output a single logit per graph (binary classification)
        return out

def train(model, loader):
    model.train()
    total_loss = 0
    y_true_train = []
    y_pred_train = []

    for batch in loader:
        x = batch.x.to(device)  # Node features
        edge_index = batch.edge_index.to(device)  # Edge index (hyperedges)
        y = batch.y.to(device).float()  # Labels (binary classification)
        batch_idx = batch.batch.to(device)  # The batch assignment for pooling

        optimizer.zero_grad()

        # Forward pass
        out = model(x, edge_index, batch_idx).view(-1)  # Get the output logits (one per graph)

        # Skip batches with missing logits (empty graphs)
        if out.size(0) != y.size(0):
            continue  # Skip this batch since it contains empty graphs

        # Compute loss
        loss = loss_fn(out, y)  # Compute loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Collect true and predicted labels for accuracy calculation
        probs = torch.sigmoid(out)
        preds = probs >= 0.5
        y_true_train.extend(y.cpu().numpy())
        y_pred_train.extend(preds.cpu().numpy())

    # Compute training accuracy
    train_accuracy = accuracy_score(y_true_train, y_pred_train)
    return total_loss / len(loader), train_accuracy

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)  # Node features
            edge_index = batch.edge_index.to(device)  # Edge index (hyperedges)
            y = batch.y.to(device).float()  # Labels
            batch_idx = batch.batch.to(device)  # Batch assignment for pooling

            # Forward pass
            out = model(x, edge_index, batch_idx).view(-1)  # Get the output logits
            probs = torch.sigmoid(out)  # Apply sigmoid to get probabilities
            preds = probs >= 0.5  # Binarize predictions

            # Collect true labels and predicted labels for accuracy
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # Compute loss
            loss = loss_fn(out, y)
            total_loss += loss.item()

    # Compute validation accuracy
    val_accuracy = accuracy_score(y_true, y_pred)
    return total_loss / len(loader), val_accuracy


def evaluate_and_plot_roc(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []  # For ROC AUC

    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)  # Node features
            edge_index = batch.edge_index.to(device)  # Edge index (hyperedges)
            y = batch.y.to(device).float()  # Labels
            batch_idx = batch.batch.to(device)  # Batch assignment for pooling

            # Forward pass
            out = model(x, edge_index, batch_idx).view(-1)  # Get the output logits
            probs = torch.sigmoid(out)  # Apply sigmoid to get probabilities
            preds = probs >= 0.5  # Binarize predictions

            # Collect true labels, predicted labels, and probabilities
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random performance)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return accuracy, precision, recall, f1, roc_auc

def main():
    model_name = "zhihan1996/DNA_bert_6"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # Load the fine-tuned model and tokenizer
    model_name = "./fine_tuned_fusion_dna_bert"  # Path to your fine-tuned model
    # Load the fine-tuned model for sequence classification
    load_model = BertForSequenceClassification.from_pretrained(model_name)


    df_fused_loaded = pd.read_csv("./df_fused_with_debruijn_edges.csv")
    #df_fused_loaded['debruijn_edges'] = df_fused_loaded['debruijn_edges'].apply(lambda x: set(ast.literal_eval(x)))
    print(df_fused_loaded.head())
    hypergraph_results = process_dataset(df_fused_loaded)


    # Assume bert_model and tokenizer are already loaded
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare the dataset with BERT-based node features
    dataset_with_bert = prepare_hypergraph_with_bert(hypergraph_results, load_model, tokenizer, device)
    loader = DataLoader(dataset_with_bert, batch_size=32, shuffle=True)
    torch.save(dataset_with_bert, './dataset_with_bert_hyper.pt')

    # Define split ratios
    train_ratio = 0.8  # 80% of data for training
    val_ratio = 0.1    # 10% for validation
    test_ratio = 0.1   # 10% for testing

    # Calculate the number of samples for each set
    num_total = len(dataset)
    num_train = int(train_ratio * num_total)
    num_val = int(val_ratio * num_total)
    num_test = num_total - num_train - num_val

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create DataLoaders for train, val, and test sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # File to save test metrics

    file_path = 'hyper_metrics_log.txt'

    # Training loop
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        # Train and get training loss and accuracy
        train_loss, train_accuracy = train(model, train_loader)

        # Validate and get validation loss and accuracy
        val_loss, val_accuracy = evaluate(model, val_loader)

        # Print training and validation metrics for each epoch
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Final evaluation on the test set
    test_acc, test_prec, test_recall, test_f1, test_roc_auc = evaluate_and_plot_roc(model, test_loader)

    # Save the test results to the file
    with open(file_path, 'a') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Precision: {test_prec:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write(f"Test F1-Score: {test_f1:.4f}\n")
        f.write(f"Test ROC AUC: {test_roc_auc:.4f}\n")
        f.write(f"-----------------------------------")

    # Print test results to console
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test ROC AUC: {test_roc_auc:.4f}")

    # Save the model state_dict after training
    model_save_path = './hypergraph_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
