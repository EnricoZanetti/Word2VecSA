#!/usr/bin/env python

import argparse
import os
import random
import sys
import time

import matplotlib as mpl
import numpy as np

from main import neg_sampling_loss_and_gradient, sgd, skipgram, word2vec_sgd_wrapper
from utils.utils import dump

mpl.use('agg')
import matplotlib.pyplot as plt

# Ensure Python Version
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5


def load_dataset(dataset_name):
    """
    Load the specified dataset. Extend this function to add support for additional datasets.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load. Currently supports 'stanford'.

    Returns
    -------
    dataset : object
        Dataset object containing tokenized data and context generation methods.
    """
    if dataset_name.lower() == 'stanford':
        from utils.treebank import StanfordSentiment

        return StanfordSentiment()
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. Please add support for your dataset."
        )


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Train Word2Vec embeddings and visualize results.'
    )
    parser.add_argument(
        '--dataset', type=str, default='stanford', help="Dataset to use ('stanford' or custom)."
    )
    parser.add_argument(
        '--dim_vectors', type=int, default=10, help='Dimension of the word embeddings.'
    )
    parser.add_argument('--context_size', type=int, default=5, help='Context window size.')
    parser.add_argument('--learning_rate', type=float, default=0.3, help='Learning rate for SGD.')
    parser.add_argument('--iterations', type=int, default=40000, help='Number of SGD iterations.')
    parser.add_argument(
        '--print_every', type=int, default=10, help='Print loss every n iterations.'
    )
    parser.add_argument(
        '--output_dir', type=str, default='output', help='Directory to save outputs.'
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset)
    tokens = dataset.tokens()
    n_words = len(tokens)

    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize word vectors
    word_vectors = np.concatenate(
        (
            (np.random.rand(n_words, args.dim_vectors) - 0.5) / args.dim_vectors,
            np.zeros((n_words, args.dim_vectors)),
        ),
        axis=0,
    )

    # Train word vectors
    start_time = time.time()
    word_vectors = sgd(
        lambda vec: word2vec_sgd_wrapper(
            skipgram, tokens, vec, dataset, args.context_size, neg_sampling_loss_and_gradient
        ),
        word_vectors,
        args.learning_rate,
        args.iterations,
        None,
        False,
        PRINT_EVERY=args.print_every,
    )

    print('Sanity check: cost at convergence should be around or below 10')
    print(f'Training took {time.time() - start_time:.2f} seconds')

    # Concatenate input and output word vectors for visualization
    word_vectors = np.concatenate((word_vectors[:n_words, :], word_vectors[n_words:, :]), axis=0)

    # Sample words to visualize (adjustable)
    visualize_words = [
        'great',
        'cool',
        'brilliant',
        'wonderful',
        'well',
        'amazing',
        'worth',
        'sweet',
        'enjoyable',
        'boring',
        'bad',
        'dumb',
        'annoying',
        'female',
        'male',
        'queen',
        'king',
        'man',
        'woman',
        'rain',
        'snow',
        'hail',
        'coffee',
        'tea',
    ]

    visualize_idx = [tokens[word] for word in visualize_words]
    visualize_vecs = word_vectors[visualize_idx, :]

    # Save word vectors for evaluation
    sample_vectors = {word: list(vec) for word, vec in zip(visualize_words, visualize_vecs)}
    sample_vectors_path = os.path.join(output_dir, 'sample_vectors_(soln).json')
    dump(sample_vectors, sample_vectors_path)

    # Dimensionality reduction via PCA (SVD)
    temp = visualize_vecs - np.mean(visualize_vecs, axis=0)
    covariance = 1.0 / len(visualize_idx) * temp.T.dot(temp)
    U, S, V = np.linalg.svd(covariance)
    coord = temp.dot(U[:, 0:2])

    for i in range(len(visualize_words)):
        plt.text(
            coord[i, 0], coord[i, 1], visualize_words[i], bbox=dict(facecolor='green', alpha=0.1)
        )

    plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
    plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

    # Save the plot
    output_plot_path = os.path.join(output_dir, 'word_vectors_(soln).png')
    plt.savefig(output_plot_path)


if __name__ == '__main__':
    main()
