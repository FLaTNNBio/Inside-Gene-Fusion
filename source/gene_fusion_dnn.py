from sklearn.metrics import roc_auc_score, roc_curve, auc,accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import logging

# Definizione di un modello di esempio per la DNN (qui andrebbe il tuo modello personalizzato)
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

# Esempio di utilizzo di TrainingArguments (puoi personalizzare i parametri)
class TrainingArguments:
    def __init__(self, output_dir, num_train_epochs, per_device_train_batch_size, per_device_eval_batch_size, warmup_steps, weight_decay, logging_dir, logging_steps, eval_strategy, save_steps):
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps
        self.eval_strategy = eval_strategy
        self.save_steps = save_steps

# Funzione di calcolo delle metriche
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

# Funzione per calcolare e salvare la curva ROC binaria
def plot_roc_binary(all_labels, all_probs, output_path="roc_curve_binary.png"):
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    logging.info(f"Test Set ROC AUC (binary): {roc_auc:.4f}")

    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Binary)')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    logging.info(f"Curva ROC salvata come '{output_path}'")

# Funzione per calcolare e salvare la curva ROC multi-classe
def plot_roc_multiclass(all_labels, all_probs, num_classes, output_path="roc_curve_multiclass.png"):
    all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    roc_auc = roc_auc_score(all_labels_bin, all_probs, multi_class='ovr')
    logging.info(f"Test Set ROC AUC (multi-class): {roc_auc:.4f}")

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green']
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Multi-class)')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    logging.info(f"Curva ROC salvata come '{output_path}'")

# Funzione per calcolare e loggare tutte le metriche
def log_test_metrics(all_preds, all_labels, num_classes):
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Loggare tutte le metriche su una sola riga
    logging.info(f"Test Set Metrics: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")


# Funzione per addestrare la DNN
def train_dnn(model, train_dataset, val_dataset, test_dataset, training_args, criterion, optimizer, num_classes):
    # Caricare i dati in DataLoader per batch processing
    train_loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_args.per_device_eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size)

    # Iniziare l'addestramento
    for epoch in range(training_args.num_train_epochs):
        model.train()
        total_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % training_args.logging_steps == 0:
                logging.info(f"Epoch [{epoch+1}/{training_args.num_train_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        if training_args.eval_strategy == "epoch":
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            metrics = compute_metrics(all_preds, all_labels)
            logging.info(f"Epoch [{epoch+1}/{training_args.num_train_epochs}] Validation Metrics: {metrics}")

        if (epoch + 1) % training_args.save_steps == 0:
            checkpoint_path = f"{training_args.output_dir}/model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Modello salvato dopo l'epoca {epoch+1} come {checkpoint_path}")

    # Valutazione finale sul test set
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

    # Log delle metriche finali sul test set
    logging.info("***** Final Test Metrics *****")
    log_test_metrics(all_preds, all_labels, num_classes)

    output_dir_roc=f"{training_args.output_dir}/ROC_embeddings_DNN"
    # Seleziona la funzione per la curva ROC in base al numero di classi
    if num_classes == 2:
        plot_roc_binary(all_labels, all_probs, output_dir_roc)
    else:
        plot_roc_multiclass(all_labels, all_probs, num_classes, output_dir_rocr)

    # Salvataggio finale del modello
    torch.save(model.state_dict(), f"{training_args.output_dir}/final_model_DNN.pth")
    logging.info(f"Model saved in {training_args.output_dir}/final_model_DNN.pth")

def main():

    # Imposta il logging per salvare su file
    logging.basicConfig(
        filename="./logs.log" ,  # Nome del file di log
        filemode='a',
        level=logging.INFO,            # Livello di logging
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Carica il file con embeddings e labels
    data = torch.load('embeddings_with_labels.pt')

    # Estrai embeddings e labels
    embeddings = data['embeddings']
    labels = data['labels']

    # Converti embeddings e labels in tensori (se non lo sono già)
    embeddings = torch.tensor(embeddings)
    labels = torch.tensor(labels)

    # Fase 1: Dividi il dataset in training+validation (80%) e test set (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        embeddings, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,  # Mantiene la distribuzione delle classi
        shuffle=True      # Shuffle esplicito
    )

    # Fase 2: Dividi il training+validation set in training set (80% del training) e validation set (20% del training)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.2,   # 20% del training+validation diventa il validation set
        random_state=42,
        stratify=y_train_val,  # Mantiene la distribuzione delle classi
        shuffle=True           # Shuffle esplicito
    )

    print(f"Dimensioni Training set: {X_train.shape[0]}")
    print(f"Dimensioni Validation set: {X_val.shape[0]}")
    print(f"Dimensioni Test set: {X_test.shape[0]}")

    # Inizializza il modello con i parametri corretti (input, hidden e output size)
    input_size = X_train.shape[1]  # Numero di feature negli embeddings
    hidden_size = 128  # Puoi cambiare questo valore
    output_size = len(torch.unique(labels))  # Numero di classi uniche
    model = SimpleDNN(input_size, hidden_size, output_size)

    # Definizione della loss e dell'optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    training_args = TrainingArguments(
        output_dir='./',           # Directory di output
        num_train_epochs=50,              # Numero di epoche
        per_device_train_batch_size=32,   # Batch size per il training
        per_device_eval_batch_size=32,    # Batch size per la validazione
        warmup_steps=500,                 # Warmup steps per il learning rate scheduler
        weight_decay=0.01,                # Weight decay
        logging_dir='./logs',             # Directory per i log
        logging_steps=10,                 # Log ogni 10 passi
        eval_strategy="epoch",            # Valutazione ogni epoca
        save_steps=5                      # Salva ogni 5 epoche
    )

    # Trasforma i dati in TensorDataset
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    num_classes = 2

    logging.shutdown()  # Flush immediato dei log
    logging.info("Start Trainig...")
    # Addestramento del modello
    train_dnn(model, train_dataset, val_dataset, test_dataset, training_args, criterion, optimizer, num_classes)

if __name__ == "__main__":
    main()