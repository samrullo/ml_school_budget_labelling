from warnings import warn
import numpy as np
import pandas as pd


def multilabel_sample(binary_matrix_indicator_labels, size=1000, classes_min_count=5, random_seed=None):
    """Take binary indicator matrix and return indices for a sample of size `size`
        Guarantees that sample contains minimum number of `classes_min_count` in the sample for each class.
    """
    try:
        if (np.unique(binary_matrix_indicator_labels).astype(int) != np.array([0, 1])).any():
            raise ValueError("only binary indicator matrix is allowed")
    except(TypeError, ValueError):
        raise ValueError("multilabel_sample only works with binary indicator matrices")

    if (binary_matrix_indicator_labels.sum() < classes_min_count).any():
        raise ValueError("Some classes don't have enough number of samples. Consider changing min_count")
    if size < 1:
        size = np.floor(binary_matrix_indicator_labels.shape[0] * size)

    if binary_matrix_indicator_labels.shape[1] * classes_min_count > size:
        msg = """Size is less than number of classes multiplied by min_count. Will return {} number of samples instead of {}"""
        warn(msg.format(binary_matrix_indicator_labels.shape[1] * classes_min_count, size))
        size = binary_matrix_indicator_labels.shape[1] * classes_min_count

    rng = np.random.RandomState(random_seed if random_seed is not None else np.random.randint(1))

    if isinstance(binary_matrix_indicator_labels, pd.DataFrame):
        choices = binary_matrix_indicator_labels.index
        binary_matrix_indicator_labels = binary_matrix_indicator_labels.values
    else:
        choices = np.arange(binary_matrix_indicator_labels.shape[0])

    sample_idxs = np.array([], dtype=choices.dtype)

    # first guarantee minimum number of samples for each class
    for label in range(binary_matrix_indicator_labels.shape[1]):
        label_choices = choices[binary_matrix_indicator_labels[:, label] == 1]
        label_idx_sampled = rng.choice(label_choices, size=classes_min_count, replace=False)
        sample_idxs = np.concatenate([sample_idxs, label_idx_sampled])

    # in the above loop the same indices might have been added to sample_idxs, so need to remove duplicate indices
    sample_idxs = np.unique(sample_idxs)

    # now that we have minimum number of samples for each class the rest is random samples
    sample_count = int(size - len(sample_idxs))
    remaining_choices = np.setdiff1d(choices, sample_idxs)
    remaining_sampled = rng.choice(remaining_choices, size=sample_count, replace=False)
    return np.concatenate([remaining_sampled, sample_idxs])


def multilabel_sample_dataframe(df, binary_indicator_matrix, size, min_count=5, seed=None):
    """Take a dataframe and return sample of size `size`, at the same time guarantee min_count of classes in the sample"""
    idxs = multilabel_sample(binary_indicator_matrix, size=size, classes_min_count=min_count, random_seed=seed)
    return df.loc[idxs]


def multilabel_train_test_split(X, Y, size, min_count=5, seed=None):
    """
    sample based on features matrix X and binary indicator matrix of labels Y,
    guaranteeing min_count of each class in the sample
    :param X:feature matrix
    :param Y: binary indicator matrix
    :param size: sample size
    :param min_count: minimum number of each class in the sample
    :param seed: randomness seed
    :return:(X_train, X_test, Y_train, Y_test)
    """
    index = Y.index if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])
    test_set_idxs = multilabel_sample(Y, size=size, classes_min_count=min_count, random_seed=seed)
    test_set_mask = index.isin(test_set_idxs)
    train_set_mask = ~test_set_mask
    return (X[train_set_mask], X[test_set_mask], Y[train_set_mask], Y[test_set_mask])
