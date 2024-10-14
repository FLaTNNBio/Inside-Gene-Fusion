import os
import torch
import csv
import pandas as pd
from csv import reader
import numpy as np
from sklearn.model_selection import train_test_split
import io
import itertools

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmers = []
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    #print(kmer)
    kmer = ' '.join(kmer.replace('\n','')for kmer in kmer)
    #kmers = " ".join(kmer)
    return kmer.split()

def kmerize_seq(reads_path,read_file,kmer_file, k):
    kmers_dict = {}
    with open(os.path.join(reads_path,read_file), "r") as f:
        for index, line in enumerate(f):
            if index % 2 == 0:
                continue
            #print(line)
            kmers = seq2kmer(line, k)
            #kmers, kmers_dict = update_kmers_dict(kmers,kmers_dict)
            #print(kmers)
            with open(kmer_file, 'w') as p: # c'era una 'a' di append al posto di 'w' se non funge rimetti la 'a'
                for kmer in kmers:
                    p.write(kmer + ' ')
                p.write('\n')
            p.close()
        return kmers


import subprocess


def run_shredder(input_fastq, out_file):
    """
    Runs the `gt shredder` command with specified input and output files.

    Parameters:
    - input_fastq: Path to the input FASTQ file.
    - out_file: Path to the output file.
    """
    # Command to execute
    command = [
        'gt', 'shredder',
        '-minlength', '150',
        '-maxlength', '150',
        '-overlap', '0',
        '-clipdesc', 'no',
        input_fastq
    ]

    # Redirect output to a file
    with open(out_file, 'w') as outfile:
        # Run the command
        result = subprocess.run(command, stdout=outfile, stderr=subprocess.PIPE, text=True)

        # Check for errors
        if result.returncode != 0:
            print(f"Error occurred: {result.stderr}")
        else:
            print(f"Shredding completed, output saved to: {out_file}")

# Run the shredding process
def gt_shredder_df(fastq_files,trans_path):
    out_dir = "./source/gene-fusion-kmer-main/data/gt_shredder_150/"
    for fastq in fastq_files:
        input_fastq = os.path.join(trans_path, fastq)
        print(input_fastq)
        read_name = os.path.splitext(os.path.basename(fastq))[0] + ".reads"
        #print(read_name)
        out_file = os.path.join(out_dir, read_name)
        print(out_file)
        f = open(out_file,"w")
        f.close()
        #run_shredder(input_fastq, out_file)
        !gt shredder -minlength 150 -maxlength 150 -overlap 0 -clipdesc no {input_fastq} > {out_file}
        for file_ext in ['.sds', '.ois', '.md5', '.esq', '.des', '.ssp']:
            rm_file = trans_path + os.path.splitext(fastq)[0] + ".fastq" + file_ext
            #print(rm_file)
            os.remove(rm_file)

def create_kmers(reads_path,kmers_path, k):
    #for reads files in reads folder
    reads_files = os.listdir(reads_path)
    count_gene = 0
    for read_file in reads_files:
        kmer_file = os.path.splitext(os.path.basename(os.path.join(reads_path, read_file)))[0] + "_kmer" + str(k) +".txt"
        kmer_path_file = os.path.join(kmers_path, kmer_file)
        print(kmer_path_file)
        kmers = kmerize_seq(reads_path,read_file,kmer_path_file, k)

def set_df_kmers(kmers_path):
    gene_count = 0
    kmers_files = os.listdir(kmers_path)
    df = pd.DataFrame(columns=['sequence', 'label'])
    for kmer_file in kmers_files:
        f = open(os.path.join(kmers_path,kmer_file),'r')
        for line in f:
            #sequence = line.strip('"')
            sequence = line.replace('\n','')
            #vedere padding sequence perchè lo dovrebbe fare gia il DNABERT di base
            #padded_sequence = sequence.center(739, "0")
            # creare le sentence direttamente da qua
            df.loc[len(df.index)] = [sequence, gene_count]
        gene_count += 1
    return df


def create_trigrams_string(sentence):
    kmers = sentence.split()
    trigrams = [kmers[i:i+3] for i in range(len(kmers)-2)]
    for trigram in trigrams:
        yield " ".join(trigram)


def create_tr_dataset(kmer_dataset, path_to_save_dataset):
    with open(os.path.join(path_to_save_dataset, "sentences.csv"), "w", newline='') as csvfile:
        fieldnames = ["sentence", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Cicla su ogni riga del dataset
        for index, row in kmer_dataset.iterrows():
            sentence = row['sequence']
            label = row['label']

            for trigram in create_trigrams_string(sentence):
                row = {'sentence': trigram, 'label':label}#the label has to specified if the nword is fused or not
                writer.writerow(row)

def main():
    dataset_path = "./source/gene-fusion-kmer-main/dataset/"
    kmers_path = "./source/gene-fusion-kmer-main/data/kmers_6/"
    trans_path = "./source/gene-fusion-kmer-main/data/transcripts/"
    reads_path = "./source/gene-fusion-kmer-main/data/gt_shredder_150/"
    n_word = 20
    k = 6
    fastq_files = os.listdir(trans_path)
    print(fastq_files)

    gt_shredder_df(fastq_files, trans_path)
    create_kmers(reads_path,kmers_path,k)
    df_kmers = set_df_kmers(kmers_path)
    print(df_kmers.head())
    create_tr_dataset(df_kmers,dataset_path)
    print("csv salvato in ",dataset_path, "sentences.csv")

if __name__ == "__main__":
    main()
