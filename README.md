# Inside Gene Fusion

The proposed tool enables the analysis of DNA sequences using three alignment-free techniques, combining text-based and Graph Learning approaches in one tool. The tool integrates state-of-the-art methods to classify DNA sequences by leveraging deep learning models.

In order to use the proposed tools, the requiremnets are needed.
```bash
pip installpip install -r requirements.txt
```

## Text-Based Tool
### Key Features
- **Gene Classification**: Fine-tune a pre-trained DNABERT model to classify the DNA sequence to its corresponding gene.
- **Chimeric Sequence Detection**: Utilize DNABERT embedding representations to train a deep learning model to classify DNA sequences as either chimeric or non-chimeric.

## Data Preparation

To run the DNABERT model, the data must be prepared by using a specific pre-processing script.

### Pre-process the Data for Gene Classification

Run the following command to prepare your data for the DNABERT model:

```bash
python3 gene_classifier_pre_process_data_filter.py
```

The prepocess scripts are needed to train and/or use the pre-trained models. 
