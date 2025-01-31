import pandas as pd
import os
import random


def sample_metaset(input_file, output_file, num_instances=2010, random_seed=42):
    """
    Samples a total of 2010 instances from a metaset while preserving triplet structure.

    :param input_file: Path to the metaset CSV file.
    :param output_file: Path to save the sampled subset.
    :param num_instances: Total number of instances
    :param random_seed: Random seed for reproducibility.
    """
    # Load the metaset
    df = pd.read_csv(input_file)

    # Ensure the dataset contains the correct columns
    if "premise" not in df.columns or "hypothesis" not in df.columns or "label" not in df.columns:
        raise ValueError("Input file must contain 'premise', 'hypothesis', and 'label' columns.")

    # Get unique premises
    unique_premises = df["premise"].unique()

    # Calculate the number of premise sets needed to reach num_instances
    num_premise_sets = num_instances // 3  # Since each premise appears exactly three times

    # Ensure we don't sample more than available premises
    if num_premise_sets > len(unique_premises):
        raise ValueError(f"Requested {num_premise_sets} premise sets, but only {len(unique_premises)} available.")

    # Set random seed and sample premise sets
    random.seed(random_seed)
    sampled_premises = random.sample(list(unique_premises), num_premise_sets)

    # Filter the dataframe to only include selected premises (ensuring triplets are included)
    sampled_df = df[df["premise"].isin(sampled_premises)]

    # Save the sampled subset
    sampled_df.to_csv(output_file, index=False)
    print(f"Sampled subset saved to {output_file} with {len(sampled_df)} rows.")


input_folder = "../data/processed"
output_folder = "data"
os.makedirs(output_folder, exist_ok=True)

for i in range(5):  # Iterate over metaset_0 to metaset_4
    input_metaset = os.path.join(input_folder, f"metaset_{i}.csv")
    output_subset = os.path.join(output_folder, f"metaset_{i}_subset.csv")
    sample_metaset(input_metaset, output_subset)
