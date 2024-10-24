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


import glob

def process_fq_file(file_path, output_fq):
    with open(file_path, 'r') as fq_file, open(output_fq, 'a') as new_fq_file:
        # Read the fq file in chunks of 4 lines
        while True:
            ref_line = fq_file.readline().strip()  # @ref line
            seq_line = fq_file.readline().strip()  # sequence line
            plus_line = fq_file.readline().strip()  # discard the '+' line
            qual_line = fq_file.readline().strip()  # discard the quality line
            
            # Break the loop if the file ends
            if not ref_line or not seq_line:
                break
            
            # Write only the ref_line and seq_line to the new FASTQ file, plus the required '+' and dummy quality line
            new_fq_file.write(f"{ref_line}\n")
            new_fq_file.write(f"{seq_line}\n")
            new_fq_file.write(f"{plus_line}\n")
            new_fq_file.write(f"{qual_line}\n")

def process_all_fq_files(input_folder, output_fq):
    # Clear the FASTQ file if it already exists
    with open(output_fq, 'w') as new_fq_file:
        pass  # Just opening it in write mode to clear the contents
    
    # Get all .fq files in the input folder
    fq_files = glob.glob(f"{input_folder}/*.fq")
    
    # Process each .fq file
    for fq_file in fq_files:
        process_fq_file(fq_file, output_fq)


def main():
    
    fastq_directory = "./source/gene-fusion-kmer-main/data/transcripts/"  # Directory containing the FASTQ files
    art_folder = "./source/gene-fusion-kmer-main/data/art_output/"
    dtnc = "./source/gene-fusion-kmer-main/dataset/dataset_non_chimeric.fastq"
    dtc = "./source/gene-fusion-kmer-main/dataset/dataset_chimeric.fastq"


    # Get all FASTQ files in the specified directory
    fastq_files = [os.path.join(fastq_directory, f) for f in os.listdir(fastq_directory) if f.endswith('.fastq')]

    if not fastq_files:
        print(f"No FASTQ files found in directory {fastq_directory}")
        sys.exit(1)
    
    # Run the merging and replacement
    create_non_chimeric_dataset(fastq_files, dtnc)
    process_all_fq_files(art_folder, dtc)

if __name__ == "__main__":
    main()
