import os
import sys

def create_non_chimeric_dataset(input_files, output_file):
    with open(output_file, 'w') as outfile:
        for file in input_files:
            with open(file, 'r') as infile:
                for line in infile:
                    if line.startswith('>'):
                        # Replace '>' with '@' at the start of the line
                        outfile.write('@' + line[1:])
                    else:
                        outfile.write(line)

def main():
    
    fastq_directory = "./source/gene-fusion-kmer-main/data/transcripts/"  # Directory containing the FASTQ files
    dtc = "./source/gene-fusion-kmer-main/dataset/dataset_non_chimeric.fastq"


    # Get all FASTQ files in the specified directory
    fastq_files = [os.path.join(fastq_directory, f) for f in os.listdir(fastq_directory) if f.endswith('.fastq')]

    if not fastq_files:
        print(f"No FASTQ files found in directory {fastq_directory}")
        sys.exit(1)
    
    # Run the merging and replacement
    create_non_chimeric_dataset(fastq_files, dtc)

if __name__ == "__main__":
    main()
