"""ML related data util functions

"""
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_validate(data, train_size=0.8, shuffle=False, seed=None):
    assert train_size <= 1 and train_size > 0
    data.reset_index(drop=True, inplace=True)
    
    if train_size == 1:  # it means all samples go to train
        val_data = None
        train_data = data.copy()
        if shuffle:  # Shuffle all rows
            train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        train_idx, val_idx = train_test_split(data.index, train_size=train_size,
                                              shuffle=shuffle, random_state=seed)
        train_data = data.loc[train_idx].reset_index(drop=True)
        val_data = data.loc[val_idx].reset_index(drop=True)

    return({"train": train_data,
            "validate": val_data})


def down_sample(data, label_col, pos_size=0.5, seed=None):
    """Down-sample the negative samples in the original imbalanced data

    :param data:
    :param label_col:
    :param pos_size: Size (percentage) of the positive samples desired
    :param seed:
    :return:
    """
    pos_data = data[data[label_col] == 1].copy()  # positive samples
    pos_length = len(pos_data)

    neg_data = data[data[label_col] == 0].copy()  # negative samples
    neg_length = (1.0 - pos_size) / pos_size * pos_length

    neg_data = neg_data.sample(n=neg_length, replace=False, random_state=seed)

    data = pd.concat([pos_data, neg_data]).sample(frac=1, replace=False).reset_index()
    return data


def up_sample_naive(data, label_col, pos_size = 0.5, seed=None):
    """Up-sample the postive samples in the original imbalanced data using naive method

    :param data:
    :param label_col:
    :param pos_size: Size (percentage) of the positive samples desired
    :param seed:
    :return:
    """
    neg_length = sum(data[label_col] == 0)

    pos_length = pos_size / (1.0 - pos_size) * neg_length
    pos_data = data[data[label_col] == 1].copy()  # positive samples
    assert pos_length >= len(pos_data), "Required positive sample length cannot be smaller than the current"

    pos_data_new = pos_data.sample(n=pos_length - len(pos_data), replace=True)
    data = pd.concat([data, pos_data_new]).sample(frac=1, replace=False).reset_index()
    return data
