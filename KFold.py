import pandas
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

X = pd.read_csv('train2017.txt', sep="\n", header=None).append(pd.read_csv('val2017.txt', sep="\n", header=None), ignore_index=True)

n_splits = 3
splitter = ShuffleSplit(n_splits=n_splits, test_size=1.0/n_splits, random_state=42)

for fold_i, (train_index, test_index) in enumerate(splitter.split(X)):
    folded_train, folded_val = X.iloc[train_index], X.iloc[test_index]
    folded_train.to_csv(f'train2017_fold{fold_i}.txt', index=False, header=False)
    folded_val.to_csv(f'val2017_fold{fold_i}.txt', index=False, header=False)