import os
import subprocess
from tqdm import tqdm

def run_art_illumina(fasta_filename, fusim_fasta_dir_path, art_output_path):
    subprocess.run([
        'art_illumina',
        '-i', os.path.join(fusim_fasta_dir_path, fasta_filename),
        '-l', '150',
        '-f', '10',
        '-p',
        '-m', '400',
        '-s', '10',
        '-o', os.path.join(art_output_path, fasta_filename)
    ], capture_output=False, )

if __name__ == '__main__':
    data_path = "./source/gene-fusion-kmer-main/data/fusim_data"
    fusim_output_path = os.path.join(data_path, "fusim_output")
    fusim_fasta_dir_path = os.path.join(fusim_output_path, "fusim_fasta_dir")
    art_output_path = os.path.join(data_path, "art_output")
    os.makedirs(art_output_path, exist_ok=True)

    fasta_files = os.listdir(fusim_fasta_dir_path)
    for fasta_filename in fasta_files:
        run_art_illumina(fasta_filename, fusim_fasta_dir_path, art_output_path)
