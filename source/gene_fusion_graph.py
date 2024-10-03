import torch
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
from transformers import BertTokenizer,BertForSequenceClassification
import ast

def get_feature_matrix(nodes, model, device, batch_size=32):
    model.eval()  # Set BERT model to evaluation mode
    features_matrix = []
    tokenizer = BertTokenizer.from_pretrained('zhihan1996/DNA_bert_6')

    # Batch processing to speed up tokenization and inference
    for i in range(0, len(nodes), batch_size):
        batch_nodes = nodes[i:i+batch_size]

        # Tokenize the batch of nodes
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

    return features_matrix




def create_graph_data(df, model, use_gpu=True):
    data_list = []

    # Check if GPU is available and move the model to GPU if necessary
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Move BERT model to GPU if available

    for index, row in df.iterrows():
        edges = row['debruijn_edges']  # Use precomputed De Bruijn edges
        label = row['label']  # Use the label directly

        # Step 1: Create node set and edge index
        nodes = list(set([node for edge in edges for node in edge]))  # Ensure nodes are ordered

        # Create a mapping from nodes to indices
        node_index = {node: i for i, node in enumerate(nodes)}

        # Create edge indices by mapping nodes to their integer indices
        edge_index = [[node_index[edge[0]], node_index[edge[1]]] for edge in edges]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)  # Move edge_index to GPU

        # Step 2: Create a feature matrix for nodes using BERT embeddings
        features_matrix = get_feature_matrix(nodes, model, device)  # Pass device to get BERT embeddings
        features_matrix = torch.tensor(features_matrix, dtype=torch.float).to(device)  # Move features_matrix to GPU

        # Step 3: Create label tensor (for binary classification, chimeric or non-chimeric)
        y = torch.tensor([label], dtype=torch.long).to(device)  # Move label tensor to GPU

        # Step 4: Create a graph data object
        data = Data(x=features_matrix, edge_index=edge_index, y=y)
        data_list.append(data)

        print(f'{index+1:4d} Graph created: ')
        print(data)

    return data_list


def save_graph_data(data_list, filename):
    # Save the data_list to a file
    torch.save(data_list, filename)
    print(f"Graph data saved to {filename}")



def load_graph_data(filename):
    # Load the data_list from a file
    data_list = torch.load(filename)
    print(f"Graph data loaded from {filename}")
    return data_list



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
        # Unpack the data object
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply the first GCN layer followed by ReLU and dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout

        # Apply the second GCN layer followed by ReLU and dropout
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout again

        # Perform global mean pooling over all nodes in the graph
        x = global_mean_pool(x, batch)  # Pooling to get graph-level representation

        # Apply the final linear layer and return the raw logits (no activation here)
        x = self.fc(x)

        return x  # Return raw logits


# Training function
def train():
    model.train()  # Set the model to training mode
    total_loss = 0  # Initialize total loss for the epoch

    for data in train_loader:
        data = data.to(device)  # Move the batch data to the GPU (or CPU)
        optimizer.zero_grad()  # Zero the gradients from the previous step

        # Forward pass: Compute model output
        out = model(data).squeeze()  # Get logits and remove singleton dimension if needed

        # Make sure output and target labels match in size
        min_size = min(out.size(0), data.y.size(0))

        # If there's a mismatch in the batch size, slice both the output and target to the minimum size
        if out.size(0) != data.y.size(0):
            out = out[:min_size]  # Slice output to match the size of data.y
            data.y = data.y[:min_size]  # Slice the labels to match the size of the output

        # Compute the loss (with the output and labels both having the same shape now)
        loss = loss_fn(out, data.y.float())  # Ensure data.y is float for BCEWithLogitsLoss
        loss.backward()  # Backpropagate the loss to compute gradients
        optimizer.step()  # Perform an optimizer step (update weights)

        total_loss += loss.item()  # Accumulate the loss for this batch

    return total_loss / len(train_loader)  # Return the average loss over all batches


def evaluate(loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0  # Initialize counter for correct predictions
    total = 0  # Initialize counter for total samples

    for data in loader:
        data = data.to(device)  # Move batch data to the device (GPU/CPU)

        with torch.no_grad():  # Disable gradient computation for evaluation
            out = model(data).squeeze()  # Get logits and remove singleton dimension if needed

            # Ensure output and labels match in size
            min_size = min(out.size(0), data.y.size(0))
            if out.size(0) != data.y.size(0):
                out = out[:min_size]  # Slice output to match size of labels
                data.y = data.y[:min_size]  # Slice labels to match size of output

            # Apply sigmoid to get probabilities, and classify as 1 if probability >= 0.5
            preds = torch.sigmoid(out) >= 0.5

            # Ensure preds and labels are of the same shape
            labels = data.y.view(-1).float()  # Ensure labels are of shape (batch_size,) and float

            # Calculate number of correct predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)  # Count total samples

    return correct / total  # Return accuracy


# Function to compute ROC AUC on the test set and return the predictions and labels
def compute_roc_auc(loader):
    model.eval()
    all_labels = []
    all_preds = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data).squeeze()  # Get logits
            probs = torch.sigmoid(out)  # Get probabilities using sigmoid
            all_preds.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Check if both classes are present in the test set
    if len(np.unique(all_labels)) == 1:
        print("Only one class present in y_true. ROC AUC score is not defined.")
        return None, None, None

    # Calculate ROC AUC
    roc_auc = roc_auc_score(all_labels, all_preds)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc, all_preds, all_labels

def main():
    # Load the fine-tuned model and tokenizer
    model_name = "./fine_tuned_dna_bert"  # Path to your fine-tuned model

    df_fused_loaded = pd.read_csv('./df_fused_with_debruijn_edges.csv')
    df_fused_loaded['debruijn_edges'] = df_fused_loaded['debruijn_edges'].apply(lambda x: set(ast.literal_eval(x)))
    print(df_fused_loaded.head())
    # Load the fine-tuned model for sequence classification
    load_model = BertForSequenceClassification.from_pretrained(model_name)

    graph_data_list = create_graph_data(df_fused_loaded,load_model)

    save_graph_data(graph_data_list, "graph_data_1.pth")

    # Training loop
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        train_loss = train()
        train_acc = evaluate(train_loader)
        val_acc = evaluate(val_loader)

        print(f"Epoch {epoch}, Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Final evaluation on the test set
    test_acc = evaluate(test_loader)
    test_roc_auc, test_preds, test_labels = compute_roc_auc(test_loader)

    with open("./performance_metrics_graph.txt", "a") as f:
        # Append a separator or fold identifier (optional, to distinguish runs)
        f.write("\n--- New Performance Metrics ---\n")

            # Append performance metrics to the file
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test ROC AUC: {test_roc_auc:.4f}\n")

            # Binarize the predictions using 0.5 threshold
        test_preds_binarized = (np.array(test_preds) >= 0.5).astype(int)

            # Calculate Precision, Recall, and F1-score
        precision = precision_score(test_labels, test_preds_binarized)
        recall = recall_score(test_labels, test_preds_binarized)
        f1 = f1_score(test_labels, test_preds_binarized)

            # Append the metrics to the file
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")

        # File path to save the trained model
    model_save_path = './gcn_model.pth'

    # Save the model state_dict after training
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Initialize the model (same architecture as before)
    model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model = model.to(device)

    # Load the saved model parameters
    model.load_state_dict(torch.load('gcn_model.pth'))

    # Set the model to evaluation mode if you're using it for inference
    model.eval()

if __name__ == "__main__":
    main()