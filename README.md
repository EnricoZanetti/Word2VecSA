# Word2Vec Implementation with Sentiment Analysis

`Word2VecSA` repository provides an implementation of Word2Vec for training word embeddings and applying them to a sentiment analysis task. While the Stanford Sentiment Treebank (SST) dataset is used as an example, the package is versatile and can be adapted for use with your own datasets.

Upon completing the training process, the script generates a visualization of the trained word embeddings using dimensionality reduction (PCA). Below is an example of the output:

![Word Embedding Visualization](src/output/word_vectors_(soln).png)

## What This Repository Does

1. Word2Vec Training:
   - Implements the skip-gram model to train word embeddings.
   - Supports loss functions such as naive softmax and negative sampling.

2. Sentiment Analysis Task:
   - Demonstrates the use of trained word embeddings for analyzing sentiment using the SST dataset as an example.

3. Word Embedding Visualization:
   - Reduces the dimensionality of word embeddings to enable visualization of their relationships.

---

## Installation

### Using Conda
This package is distributed as a Conda environment. To install:

```bash
conda env create -f environment.yml
conda activate Word2VecSA
```

### Editable Installation

Important, install this package as editable:

```bash
pip install -e .
```

This enables you to modify the source code and immediately use the changes without reinstallation.


## How to Run the Code

### Step 1: Prepare Your Dataset
- To use the Stanford Sentiment Treebank (SST) dataset (default), no additional setup is required.
- If you have a custom dataset, ensure it is preprocessed into a compatible format (e.g., tokenized sentences and context windows). Update the load_dataset() function in run.py to handle your dataset or ensure it’s placed in the src/utils/datasets folder for integration.

### Step 2: Execute the Training Script
Run the following command to train the Word2Vec model and execute the sentiment analysis task:
```bash
python run.py --dataset <dataset_name> --dim_vectors <dimension> --context_size <size> --learning_rate <rate> --iterations <iterations>
```
#### Example (using SST Dataset)
```bash
python run.py --dataset stanford --dim_vectors 10 --context_size 5 --learning_rate 0.3 --iterations 40000
```
#### Example (Custom Dataset)
Replace <dataset_name> with the name of your dataset. Ensure your dataset is supported in `load_dataset()` in `run.py`.

#### **Key Parameters**

The following parameters can be customized via the command line when running `run.py`:

| **Parameter**     | **Description**                                          | **Default**     |
|--------------------|----------------------------------------------------------|-----------------|
| `--dataset`        | Name of the dataset to use (e.g., `stanford` or custom). | `stanford`      |
| `--dim_vectors`    | Dimension of word embeddings.                            | `10`            |
| `--context_size`   | Context window size.                                     | `5`             |
| `--learning_rate`  | Learning rate for SGD.                                   | `0.3`           |
| `--iterations`     | Number of iterations for SGD.                            | `40000`         |
| `--output_dir`     | Directory to save outputs.                               | `output`        |


### Step 3: Training Process
- The script will train word embeddings using the specified number of iterations (default: 40,000) of Stochastic Gradient Descent (SGD).
- Training duration may vary depending on your system and the efficiency of your implementation.

### Step 4: Output

Upon completion:
1. A visualization of the word embeddings will be displayed and saved as:
	- `word_vectors.png` in the `output/` directory.
2.	Corresponding word vectors will be saved as:
	- `sample_vectors_(soln).json` in the `output/` directory.

## Notes
- Versatility: The script is designed to work with various datasets. Add custom datasets by modifying `load_dataset()` in `run.py`.
- Performance: Training time may vary depending on system performance and dataset size.
- Outputs: Ensure the `output/` directory exists (or specify another directory) to save results (`word_vectors.png` and `sample_vectors_(soln).json`).

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project was inspired by [Natural Language Processing with Deep Learning](https://online.stanford.edu/courses/xcs224n-natural-language-processing-deep-learning) course provided by Stanford University.

## Contacts

- LinkedIn: [Enrico Zanetti](https://www.linkedin.com/in/enrico-zanetti/)
