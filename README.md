# Inside Gene Fusion

The proposed tool enables the analysis of DNA sequences using three alignment-free techniques, combining text-based and Graph Learning approaches in one tool. The tool integrates state-of-the-art methods to classify DNA sequences by leveraging deep learning models.
We propose a novel DL-based model that learns to recognize the hidden patterns that allow us to identify chimeric RNAs deriving from oncogenic gene fusions. 
This consists of a double-classifier framework which first classifies the sequence of the k-mers of a read, and then infers the chimeric information by giving as input the list of k-mer classes to a transformer-based classifier

In order to use the proposed tools, the requiremnets are needed.
```bash
pip installpip install -r requirements.txt
```

## Text-Based Tool
The first model that is defined within this project is the **gene classifier** model. 
The goal of this model is to correctly classify sentences in the source gene. 
More formally, we define a sentence as a string consisting of *n* words each 
separated by space, where each word is a *kmer*.

Starting with a read, we generate all possible kmers, of length ```len_kmer```, of the read. 
Let ```n_words``` be the number of kmers that make up a sentence, then all possible subsets of consecutive 
kmers of cardinality ```n_words``` are generated. This allows all possible sentences to be generated from a 
read. The goal of the classifier is to correctly classify a sentence to the source gene of the read used 
to generate the sentence

### Key Features
- **Gene Classification**: Fine-tune a pre-trained DNABERT model to classify the DNA sequence to its corresponding gene.
- **Chimeric Sequence Detection**: Utilize DNABERT embedding representations to train a deep learning model to classify DNA sequences as either chimeric or non-chimeric.

### Data Preparation

To run the DNABERT model, the data must be prepared by using a specific pre-processing script.

#### Pre-process the Data for Gene Classification

Run the following command to prepare your data for the DNABERT model:

```bash
python3 gene_classifier_pre_process_data_filter.py
```

Feel free to modify the k-mers and n_word variables to achieve personalized goals. The preprocessing scripts are necessary to train and/or use the pre-trained models. 

### Fine-Tune DNABERT for Gene Classification
To fine-tune the pre-trained model DNABERT for Gene Classification run the following command:
```bash
python3 dnabert_geneclassifier_fine_tune.py
```
Ensure that you modify the n_labels variable to match the number of labels in your customized dataset.

### Train Fusion Model
To identify a DNA sequence as chimeric or not, a Deep Learning model is trained on the embedding representation of the sequences  

