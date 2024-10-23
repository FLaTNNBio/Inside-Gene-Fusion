import os
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def run_fusion(gene1, gene2, fusim_fasta_dir_path, fusim_txt_dir_path):
    tqdm.write(f'Starting fusion: {gene1} + {gene2}')
    try:
        result = subprocess.run(['java', '-jar', './fusim-0.2.2/fusim.jar',
            '--gene-model=./fusim-0.2.2/refFlat.txt',
            '--fusions=10',
            '--reference=./fusim-0.2.2/hg19.fa',
            f'--fasta-output={fusim_fasta_dir_path}/fusion_{gene1}_{gene2}.fasta',
            f'--text-output={fusim_txt_dir_path}/fusion_{gene1}_{gene2}.txt',
            '-1', gene1,
            '-2', gene2,
            '--cds-only',
            '--auto-correct-orientation'], capture_output=True, text=True)

        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)

        return result
    except subprocess.CalledProcessError as e:
        print(f"Error during running {gene1} + {gene2}: {e.stderr}")
        return None

if __name__ == '__main__':
    genes_panel_path = "./source/gene-fusion-kmer-main/data/genes_panel.txt"
    data_path = "./source/gene-fusion-kmer-main/data/fusim_data"
    fusim_output_path = os.path.join(data_path, "fusim_output")
    fusim_fasta_dir_path = os.path.join(fusim_output_path, "fusim_fasta_dir")
    fusim_txt_dir_path = os.path.join(fusim_output_path, "fusim_txt_dir")
    os.makedirs(fusim_fasta_dir_path, exist_ok=True)
    os.makedirs(fusim_txt_dir_path, exist_ok=True)

    genes_panel = []
    with open(genes_panel_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            genes_panel.append(line)

    total_combinations = len(genes_panel) ** 2

    with tqdm(total=total_combinations, dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor() as executor:
            futures = []
            for gene1 in genes_panel:
                for gene2 in genes_panel:
                    futures.append(executor.submit(run_fusion, gene1, gene2, fusim_fasta_dir_path, fusim_txt_dir_path))

            for future in futures:
                future.result()
                pbar.update(1)
