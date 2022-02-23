import json
import pickle
import itertools

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

def read_text_file(path, encoding="utf-8", lines=False):
    with open(path, "r", encoding=encoding) as file:
        text = file.read()
        if lines:
            text = text.split("\n")[:-1]
    return text

def write_to_text_file(text, path, encoding="utf-8", lines=False):
    with open(path, "w", encoding=encoding) as file:
        if lines:
            file.writelines(t+"\n" for t in text)
        else:
            file.write(text)

def read_json_file(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as file:
        out = json.load(file)
    return out

def write_to_json_file(obj, path, encoding="utf-8", indent=4, ensure_ascii=False, **kwargs):
    with open(path, "w", encoding=encoding) as file:
        json.dump(obj, file, indent=indent, ensure_ascii=ensure_ascii, **kwargs)

def read_pickle_file(path):
    with open(path, "rb") as file:
        out = pickle.load(file)
    return out

def write_to_pickle_file(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)

def cosine_similarity(x, y):
    """Cosine of the angle between the vectors x and y.
    
    The cosine similarity is contained in the range [-1, 1], and is equal to:
    - 1 if x and y are positively colinear;
    - 0 if x and y are orthogonal;
    - -1 if x and y are negatively colinear.
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def cosine_distance(x, y):
    """Cosine distance between the vectors x and y.
    
    The cosine distance is an inverted version of the cosine similarity, normalized to be between 0 and 1.
    It is contained in the range [0, 1], and is equal to:
    - 0 if x and y are positively colinear;
    - 1/2 if x and y are orthogonal;
    - 1 if x and y are negatively colinear.
    """
    return 1 - (cosine_similarity(x, y) + 1) / 2

def get_pairwise_metric_matrix(vectors, metric="l2", symmetric=False, labels=None, verbose=False):
    """Compute pairwise metrics in one or two lists of objects and return them as a matrix.
    
    Args:
        vectors (iterable of objects or tuple): Objects to compare.
            An iterable of objects can be a 2-D numpy array, with its rows the objects to compare.
            An iterable of objects can also be a pd.DataFrame, in which case the objects to compare are its columns,
            and the returned matrix will be a DataFrame with appropriate column and index labels.
                (unless the argument `labels` is set, in which case the labels are taken from that argument instead)
            This argument can also be a 2-tuple of two iterables of objects to compare, which can be of different lengths.
            If 2-tuple, the matrix will have the first iterable as its rows and the second as its columns.
        metric (str or callable): Metric to use.
            The provided string aliases work for numpy arrays of arbitrary shape,
            provided any two numpy arrays to compare have the same shape.
            Currently supported string aliases:
            - "l0": L0 distance (number of non-equal components)
            - "l1": L1 distance
            - "l2": L2 distance
            - "linf": L-infinity distance
            - "dot": Dot product
            - "cosine_sim": Cosine similarity (cosine of the angle between 2 vectors)
            - "cosine_dist": Cosine distance (1 - cosine similarity, normalized to be between 0 and 1)
            - "cov": Covariance (calculated with n in the denominator, i.e. the biased estimator)
            - "corr": Correlation (calculated with n in the denominator, i.e. the biased estimator)
            - "eq_sim": Proportion of times the two vectors are equal.
            - "eq_dist": Proportion of times the two vectors are not equal.
            - "bool_sim": Taking each vector as boolean, proportion of times the two vectors agree.
            - "bool_dist": Taking each vector as boolean, proportion of times the two vectors disagree.
            If callable, must take two objects as positional arguments and return a single number.
            Default: "l2".
        symmetric (bool): Whether the provided metric function is assumed to be symmetric.
            If True, will only make about half of the calculations and automatically fill the 
            If `metric` is provided as a string, this argument will be ignored and automatically inferred.
            Default: False.
        labels (list of str, tuple or None): Labels to add to the returned matrix, which will then be returned as a DataFrame
            rather than a numpy array.
            If tuple, must be of length two and be of the form (row_labels, column_labels).
            If list of strings, 
            If None, return result as a numpy array unless one of the iterables of objects to compare is a DataFrame.
        verbose (bool): Whether to print progressbar.
            Default: False.
    
    Returns:
        np.array or pd.DataFrame: Pairwise metric matrix.
            If the argument `labels` is set or if one of the iterables of objects is a DataFrame,
            then this is a DataFrame. Otherwise, it is a numpy array.
    """
    # Define iterables
    if isinstance(vectors, tuple):
        rows, columns = vectors
    else:
        rows, columns = vectors, vectors

    if isinstance(labels, tuple):
        row_labels, column_labels = labels
    else:
        row_labels, column_labels = labels, labels

    if isinstance(rows, pd.DataFrame):
        if row_labels is None:
            row_labels = rows.columns.tolist()
        rows = rows.values.T
    if isinstance(columns, pd.DataFrame):
        if column_labels is None:
            column_labels = columns.columns.tolist()
        columns = columns.values.T

    # Define distance function
    metric_functions = {
        "l0": lambda x, y: np.linalg.norm(x-y, ord=0),
        "l2": lambda x, y: np.linalg.norm(x-y, ord=2),
        "l1": lambda x, y: np.linalg.norm(x-y, ord=1),
        "linf": lambda x, y: np.linalg.norm(x-y, ord=np.inf),
        "dot": np.dot,
        "cosine_sim": cosine_similarity,
        "cosine_dist": cosine_distance,
        "cov": lambda x, y: ((x - x.mean()) * (y - y.mean())).mean(),
        "corr": lambda x, y: ((x - x.mean()) * (y - y.mean())).mean() / np.sqrt(np.var(x) * np.var(y)),
        "eq_sim": lambda x, y: (x == y).mean(),
        "eq_dist": lambda x, y: (x != y).mean(),
        "bool_sim": lambda x, y: (x.astype("bool") == y.astype("bool")).mean(),
        "bool_dist": lambda x, y: (x.astype("bool") != y.astype("bool")).mean(),
    }
    symmetric_metrics = ["l0", "l2", "l1", "linf", "dot", "cosine_sim", "cosine_dist", "cov", "corr", "eq_sim", "eq_dist", "bool_sim", "bool_dist"]
    if callable(metric):
        metric_function = metric
    else:
        assert metric in metric_functions, f"Invalid metric string alias: {metric}."
        metric_function = metric_functions[metric]
        symmetric = metric in symmetric_metrics
    
    # Construct matrix
    n, m = len(rows), len(columns)
    if symmetric:
        # Construct the indices by first taking the upper triangular matrix in the square part,
        # then adding all the indices in the rest of the matrix.
        min_len, max_len = min(n, m), max(n, m)
        square_pairs = [(i, j) for i in range(min_len) for j in range(i, min_len)]
        if m <= n:
            pairs = square_pairs + list(itertools.product(range(min_len, n), range(m)))
        else:
            pairs = square_pairs + list(itertools.product(range(n), range(min_len, m)))
    else:
        pairs = itertools.product(range(n), range(m))
    matrix = np.zeros((n, m))
    
    for i, j in tqdm(pairs, disable=not verbose):
        res = metric_function(rows[i], columns[j])
        matrix[i, j] = res
        if symmetric and (i < m) and (j < n):
            matrix[j, i] = res
    
    # Transform to DF if any of the inputs was a DF
    if (row_labels, column_labels) != (None, None):
        matrix = pd.DataFrame(matrix, index=row_labels, columns=column_labels)

    return matrix

def scatter_with_annotations(coords, annotations):
    """Create a scatterplot based on some coords and adding text annotations.
    
    Args:
        coords (array-like): Coordinates to scatter and add text to.
            Each element must be 2-dimensional.
        annotations (list of strings): Texts to add to the plot.
            Must be of same length as coords.
    
    Returns:
        plt.Axes: Axes object for the plot.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.4)
    for name, vect in zip(annotations, coords):
        plt.annotate(name, vect)
    
    return ax