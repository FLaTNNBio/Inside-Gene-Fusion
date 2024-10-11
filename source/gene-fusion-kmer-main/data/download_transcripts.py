import requests
import os
import time

# Function to fetch transcripts for a gene using Ensembl API
def fetch_transcripts(gene_name):
    server = "https://rest.ensembl.org"
    ext = f"/lookup/symbol/homo_sapiens/{gene_name}?expand=1"
    
    headers = {"Content-Type": "application/json"}
    response = requests.get(server + ext, headers=headers)
    
    if not response.ok:
        print(f"Error fetching data for gene: {gene_name}")
        return None
    
    gene_data = response.json()
    
    if 'Transcript' in gene_data:
        transcripts = [transcript['id'] for transcript in gene_data['Transcript']]
        return transcripts
    else:
        print(f"No transcripts found for gene: {gene_name}")
        return []

# Function to fetch DNA sequence for a transcript
def fetch_transcript_sequence(transcript_id):
    server = "https://rest.ensembl.org"
    ext = f"/sequence/id/{transcript_id}"
    
    headers = {"Content-Type": "text/plain"}
    response = requests.get(server + ext, headers=headers)
    
    if not response.ok:
        print(f"Error fetching sequence for transcript: {transcript_id}")
        return None
    
    return response.text

# Function to read genes from a file
def read_gene_list(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file.readlines()]

# Function to create a file for each gene and save its transcript sequences
def save_transcript_sequences(gene_name, transcript_sequences):
    filename = f"{gene_name}_transcripts.txt"
    with open(filename, 'w') as file:
        for transcript_id, sequence in transcript_sequences.items():
            file.write(f">{transcript_id}\n")
            file.write(f"{sequence}\n\n")

# Main function to process each gene, fetch transcripts and their DNA sequences
def process_genes(input_file):
    gene_list = read_gene_list(input_file)

    for gene in gene_list:
        print(f"Processing gene: {gene}")
        transcripts = fetch_transcripts(gene)
        
        if transcripts:
            transcript_sequences = {}
            for transcript in transcripts:
                print(f"Fetching sequence for transcript: {transcript}")
                sequence = fetch_transcript_sequence(transcript)
                if sequence:
                    transcript_sequences[transcript] = sequence
                time.sleep(1)  # Pause to avoid overwhelming the API server
            
            if transcript_sequences:
                save_transcript_sequences(gene, transcript_sequences)
                print(f"Sequences for {gene} saved.")
        time.sleep(1)  # Pause between genes

def main():
  process_genes('gene_panel.txt' )

if __name__ == "__main__":
  main()