import os
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import StratifiedKFold

'''ds = load_dataset("stanfordnlp/snli")

os.makedirs("data", exist_ok=True)

ds["train"].to_csv("data/snli_train.csv")
ds["test"].to_csv("data/snli_test.csv")
ds["validation"].to_csv("data/snli_val.csv")'''

train_df = pd.read_csv("data/snli_train.csv")

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

folds = []
for _, fold_index in skf.split(train_df, train_df["label"]):
    fold = train_df.iloc[fold_index]
    folds.append(fold)

for i, fold in enumerate(folds):
    fold.to_csv(f"data/meta_{i}.csv", index=False)
    print(f"fold {i} saved with {len(fold)} folds")

