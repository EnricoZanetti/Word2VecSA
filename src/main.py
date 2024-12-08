#!/usr/bin/env python

"""
Word2Vec Implementation using Stochastic Gradient Descent (SGD).

This script includes functions for building and training Word2Vec models using
both naive softmax and negative sampling loss, as well as an SGD optimizer.
"""

# Save parameters periodically as a fail-safe
SAVE_PARAMS_EVERY = 5000

import glob
import os.path as op
import pickle
import random

import numpy as np

from utils.utils import softmax


def sigmoid(x):
    """
    Compute the sigmoid function.

    Parameters
    ----------
    x : float or np.ndarray
        Input scalar or numpy array.

    Returns
    -------
    float or np.ndarray
        The sigmoid of the input.
    """
    return 1 / (1 + np.exp(-x))


def naive_softmax_loss_and_gradient(center_word_vec, outside_word_idx, outside_vectors, dataset):
    """
    Compute naive softmax loss and gradients for Word2Vec models.

    Parameters
    ----------
    center_word_vec : np.ndarray
        Embedding of the center word.
    outside_word_idx : int
        Index of the outside word.
    outside_vectors : np.ndarray
        Outside word vectors for all words in the vocabulary.
    dataset : object
        Dataset object (not used in this implementation).

    Returns
    -------
    float
        Loss computed using naive softmax.
    np.ndarray
        Gradient with respect to the center word vector.
    np.ndarray
        Gradient with respect to all the outside word vectors.
    """
    scores = np.dot(outside_vectors, center_word_vec)
    y_hat = softmax(scores)
    loss = -np.log(y_hat[outside_word_idx])

    y = np.zeros_like(y_hat)
    y[outside_word_idx] = 1
    error = y_hat - y

    grad_center_vec = np.dot(outside_vectors.T, error)
    grad_outside_vecs = np.outer(error, center_word_vec)

    return loss, grad_center_vec, grad_outside_vecs


def get_negative_samples(outside_word_idx, dataset, K):
    """
    Sample K negative word indices.

    Parameters
    ----------
    outside_word_idx : int
        Index of the outside word to exclude from sampling.
    dataset : object
        Dataset object for sampling words.
    K : int
        Number of negative samples to generate.

    Returns
    -------
    list of int
        Indices of the sampled negative words.
    """
    neg_sample_word_indices = []
    for _ in range(K):
        newidx = dataset.sample_token_idx()
        while newidx == outside_word_idx:
            newidx = dataset.sample_token_idx()
        neg_sample_word_indices.append(newidx)
    return neg_sample_word_indices


def neg_sampling_loss_and_gradient(center_word_vec, outside_word_idx, outside_vectors, dataset, K=10):
    """
    Compute negative sampling loss and gradients for Word2Vec models.

    Parameters
    ----------
    center_word_vec : np.ndarray
        Embedding of the center word.
    outside_word_idx : int
        Index of the outside word.
    outside_vectors : np.ndarray
        Outside word vectors for all words in the vocabulary.
    dataset : object
        Dataset object for sampling words.
    K : int, optional
        Number of negative samples. Default is 10.

    Returns
    -------
    float
        Negative sampling loss.
    np.ndarray
        Gradient with respect to the center word vector.
    np.ndarray
        Gradient with respect to all the outside word vectors.
    """
    neg_sample_word_indices = get_negative_samples(outside_word_idx, dataset, K)
    indices = [outside_word_idx] + neg_sample_word_indices

    labels = np.array([1] + [-1 for _ in range(K)])
    vecs = outside_vectors[indices]

    t = sigmoid(vecs.dot(center_word_vec) * labels)
    loss = -np.sum(np.log(t))

    delta = labels * (t - 1)
    grad_center_vec = np.dot(delta, vecs)
    grad_outside_vecs = np.zeros_like(outside_vectors)
    for idx, vec_delta in zip(indices, delta):
        grad_outside_vecs[idx] += vec_delta * center_word_vec

    return loss, grad_center_vec, grad_outside_vecs


def skipgram(current_center_word, window_size, outside_words, word2ind, center_word_vectors, outside_vectors, dataset, word2vec_loss_and_gradient=neg_sampling_loss_and_gradient):
    """
    Implement the skip-gram model in Word2Vec.

    Parameters
    ----------
    current_center_word : str
        The center word.
    window_size : int
        Context window size.
    outside_words : list of str
        List of context words.
    word2ind : dict
        Mapping from words to their indices.
    center_word_vectors : np.ndarray
        Embeddings for the center words.
    outside_vectors : np.ndarray
        Embeddings for the outside words.
    dataset : object
        Dataset object for sampling.
    word2vec_loss_and_gradient : callable, optional
        Loss and gradient function. Default is `neg_sampling_loss_and_gradient`.

    Returns
    -------
    float
        Total loss for the skip-gram model.
    np.ndarray
        Gradient with respect to the center word vectors.
    np.ndarray
        Gradient with respect to the outside word vectors.
    """
    loss = 0.0
    grad_center_vecs = np.zeros_like(center_word_vectors)
    grad_outside_vectors = np.zeros_like(outside_vectors)

    center_word_idx = word2ind[current_center_word]
    center_word_vec = center_word_vectors[center_word_idx]

    for outside_word in outside_words:
        outside_word_idx = word2ind[outside_word]
        loss_curr, grad_center, grad_outside = word2vec_loss_and_gradient(
            center_word_vec, outside_word_idx, outside_vectors, dataset
        )
        loss += loss_curr
        grad_center_vecs[center_word_idx] += grad_center
        grad_outside_vectors += grad_outside

    return loss, grad_center_vecs, grad_outside_vectors


def word2vec_sgd_wrapper(word2vec_model, word2ind, word_vectors, dataset, window_size, word2vec_loss_and_gradient=neg_sampling_loss_and_gradient):
    """
    SGD wrapper for Word2Vec training.

    Parameters
    ----------
    word2vec_model : callable
        Word2Vec model function (e.g., `skipgram`).
    word2ind : dict
        Mapping from words to their indices.
    word_vectors : np.ndarray
        Embeddings for the words.
    dataset : object
        Dataset object for sampling.
    window_size : int
        Context window size.
    word2vec_loss_and_gradient : callable, optional
        Loss and gradient function. Default is `neg_sampling_loss_and_gradient`.

    Returns
    -------
    float
        Average loss over the batch.
    np.ndarray
        Gradients for the word vectors.
    """
    batchsize = 50
    loss = 0.0
    grad = np.zeros_like(word_vectors)
    N = word_vectors.shape[0]
    center_word_vectors = word_vectors[:N // 2]
    outside_vectors = word_vectors[N // 2:]

    for _ in range(batchsize):
        window_size_random = random.randint(1, window_size)
        center_word, context = dataset.get_random_context(window_size_random)

        loss_curr, grad_center, grad_outside = word2vec_model(
            center_word, window_size_random, context, word2ind,
            center_word_vectors, outside_vectors, dataset, word2vec_loss_and_gradient
        )
        loss += loss_curr / batchsize
        grad[:N // 2] += grad_center / batchsize
        grad[N // 2:] += grad_outside / batchsize

    return loss, grad


def load_saved_params():
    """
    Load previously saved parameters and reset iteration state.

    Returns
    -------
    int
        Starting iteration.
    np.ndarray or None
        Loaded parameters.
    object or None
        Loaded random state.
    """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter_ = int(op.splitext(op.basename(f))[0].split("_")[2])
        if iter_ > st:
            st = iter_

    if st > 0:
        params = np.load(f"saved_params_{st}.npy")
        with open(f"saved_state_{st}.pickle", "rb") as f:
            state = pickle.load(f)
        return st, params, state
    return st, None, None


def save_params(iter_, params):
    """
    Save parameters and random state.

    Parameters
    ----------
    iter_ : int
        Current iteration number.
    params : np.ndarray
        Parameters to save.
    """
    np.save(f"saved_params_{iter_}.npy", params)
    with open(f"saved_state_{iter_}.pickle", "wb") as f:
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, postprocessing=None, use_saved=False, PRINT_EVERY=10):
    """
    Perform Stochastic Gradient Descent (SGD).

    Parameters
    ----------
    f : callable
        Function to optimize. Should return loss and gradient.
    x0 : np.ndarray
        Initial parameter values.
    step : float
        Learning rate.
    iterations : int
        Number of iterations.
    postprocessing : callable, optional
        Function for postprocessing parameters. Default is None.
    use_saved : bool, optional
        Whether to resume from saved state. Default is False.
    PRINT_EVERY : int, optional
        Frequency of logging progress. Default is 10.

    Returns
    -------
    np.ndarray
        Final parameter values.
    """
    ANNEAL_EVERY = 20000

    if use_saved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)
        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0
    if postprocessing is None:
        postprocessing = lambda x: x

    exploss = None

    for iter_ in range(start_iter + 1, iterations + 1):
        loss, grad = f(x)
        x -= step * grad
        x = postprocessing(x)

        if iter_ % PRINT_EVERY == 0:
            if exploss is None:
                exploss = loss
            else:
                exploss = 0.95 * exploss + 0.05 * loss
            print(f"iter {iter_}: {exploss:.6f}")

        if iter_ % SAVE_PARAMS_EVERY == 0 and use_saved:
            save_params(iter_, x)

        if iter_ % ANNEAL_EVERY == 0:
            step *= 0.5

    return x
