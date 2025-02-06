import pandas as pd
from assigntools.LoLa.read_nli import snli_jsonl2dict
from data_to_curriculum import load_data

# read the data
random_baseline = pd.read_csv('curricula/random_baseline.csv', index_col=False)
baseline_random_0 = pd.read_csv('curricula/random_subset_0.csv', index_col=False)
baseline_random_1 = pd.read_csv('curricula/random_subset_1.csv', index_col=False)
baseline_random_2 = pd.read_csv('curricula/random_subset_2.csv', index_col=False)
baseline_random_3 = pd.read_csv('curricula/random_subset_3.csv', index_col=False)
baseline_random_4 = pd.read_csv('curricula/random_subset_4.csv', index_col=False)

# label disctribution
print('Random baseline:\n', random_baseline['label'].value_counts())
print('Baseline subset 0:\n',baseline_random_0['label'].value_counts()) # randomly selected baseline curricula
print('Baseline subset 1:\n',baseline_random_1['label'].value_counts())
print('Baseline subset 2:\n',baseline_random_2['label'].value_counts())
print('Baseline subset 3:\n',baseline_random_3['label'].value_counts())
print('Baseline subset 4:\n',baseline_random_4['label'].value_counts())

# 