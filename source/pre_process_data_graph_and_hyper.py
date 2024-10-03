import pandas as pd
def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

# Function to process the FASTQ file and create a DataFrame
def process_fastq(file_path, label, k):
    """
    Reads the FASTQ file and processes only the even-numbered rows (2nd, 4th, 6th, etc.)
    containing sequences. The sequences are then passed to the seq2kmer function.

    Args:
    - file_path (str): Path to the FASTQ file.
    - label (str): Label to assign to the sequences (e.g., 'chimeric' or 'non-chimeric').
    - k (int): The size of k-mers to generate.

    Returns:
    - pd.DataFrame: DataFrame containing kmerized sequences and their associated label.
    """
    sequences = []
    line_count = 0

    # Open the FASTQ file
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove any extra whitespace

            if line_count % 2 == 1:
                sequences.append(line)

            line_count += 1 # Increment the line count
        print(line_count)

    # Apply seq2kmer to each sequence
    kmerized_sequences = [seq2kmer(seq, k) for seq in sequences]

    # Create a DataFrame
    df = pd.DataFrame({
        'kmerized_sequence': kmerized_sequences,
        'label': [label] * len(kmerized_sequences)
    })

    return df

def get_debruijn_edges(kmers):
    edges = set()
    for k1 in kmers:
        for k2 in kmers:
            if k1 != k2:
                if k1[1:] == k2[:-1]:
                    edges.add((k1, k2))
                if k1[:-1] == k2[1:]:
                    edges.add((k2, k1))
    return edges


# Define a function to apply to each row
def process_kmer_sequence(sequence):
    # Convert the space-separated kmerized sequence into a list of k-mers
    kmers = sequence.split()  # Adjust this if the delimiter is different
    # Apply the get_debruijn_edges function to the list of k-mers
    return get_debruijn_edges(kmers)

def main():
    k = 6  # Set the k-mer size (you can change this value)

    # Process the FASTQ file and generate the DataFrame
    df_ch = process_fastq('dataset_chimeric2.fastq', 1, k)
    df_no_ch = process_fastq('dataset_non_chimeric.fastq', 0, k)
    df_fused = pd.concat([df_ch, df_no_ch], ignore_index=True)
    print(df_fused.head())
    df_fused.to_csv('./df_fused.csv', index=False)

    df_fused['debruijn_edges'] = df_fused['kmerized_sequence'].apply(process_kmer_sequence)
    df_fused.to_csv('./df_fused_with_debruijn_edges.csv', index=False)

if __name__ == "__main__":
    main()