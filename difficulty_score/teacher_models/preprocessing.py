import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Load the dataset
train_df = pd.read_csv("data/raw/snli_train.csv")

# Group by premise and create a representative label for each group
train_df["group_id"] = train_df.groupby("premise").ngroup()  # Assign a unique group ID for each premise
grouped_df = train_df.groupby("group_id").first().reset_index()  # Take one representative row per premise for stratification

# StratifiedKFold on grouped data
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

folds = []
for _, group_index in skf.split(grouped_df, grouped_df["label"]):
    # Select all rows belonging to the selected groups
    group_ids = grouped_df.iloc[group_index]["group_id"]
    fold = train_df[train_df["group_id"].isin(group_ids)]
    folds.append(fold)

# Save the folds to meta_*.csv files
os.makedirs("data", exist_ok=True)
for i, fold in enumerate(folds):
    fold.drop(columns=["group_id"], inplace=True)  # Remove the helper column
    fold.to_csv(f"cross_review/data/metaset_{i}.csv", index=False)
    print(f"Fold {i} saved with {len(fold)} samples.")
