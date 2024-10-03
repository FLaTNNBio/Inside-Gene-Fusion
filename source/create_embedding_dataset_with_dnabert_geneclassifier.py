from transformers import BertTokenizer, BertForSequenceClassification
import torch

def load_dataset(file_path, label):
    sequences = []
    labels = []

    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith('@'):  # Salta righe con metadati
                sequence = line.strip()
                # Se necessario, kmerizza la sequenza (ignora se già kmerizzata)
                kmerized_sequence = kmerize_sequence(sequence)
                sequences.append(kmerized_sequence)
                labels.append(label)

    return sequences, labels

def kmerize_sequence(sequence, k=6):
    return " ".join([sequence[i:i+k] for i in range(0, len(sequence), k)])

# Funzione per ottenere l'embedding di una singola sequenza
def get_embedding(sequence,model,tokenizer):
    inputs = tokenizer(sequence, return_tensors="pt", padding='max_length', truncation=True, max_length=100)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]
    # Calcola la media lungo la dimensione della sequenza (batch_size, sequence_length, hidden_size)
    sequence_embedding = last_hidden_state.mean(dim=1)
    return sequence_embedding.squeeze()

def main(dtc,dtnc):

    model_path = './multi_gene_classificator_model'
    tokenizer = BertTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
    model = BertForSequenceClassification.from_pretrained(model_path)

    chimeric_sequences, chimeric_labels = load_dataset(dtc, label=1)
    non_chimeric_sequences, non_chimeric_labels = load_dataset(dtnc, label=0)

    all_sequences = chimeric_sequences + non_chimeric_sequences
    all_labels = chimeric_labels + non_chimeric_labels
    embeddings_list = []
    labels_list = []

    for i, sequence in enumerate(all_sequences):
        embedding = get_embedding(sequence,model,tokenizer)  # Genera embedding
        label = all_labels[i]  # Ottieni la label corrispondente

        embeddings_list.append(embedding)
        labels_list.append(label)

    embeddings_tensor = torch.stack(embeddings_list)
    labels_tensor = torch.tensor(labels_list)

    # Salva embeddings e label in un file .pt
    torch.save({'embeddings': embeddings_tensor, 'labels': labels_tensor}, 'embeddings_with_labels.pt')

    print("Embeddings e label salvati nel file embeddings_with_labels.pt.")

if __name__ == "__main__":
    dtc = "./dataset_chimeric2.fastq"
    dtnc = "./dataset_non_chimeric.fastq"
    main(dtc,dtnc)