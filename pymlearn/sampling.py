"""ML related data util functions

"""
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
