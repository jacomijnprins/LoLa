import pandas as pd
import random
from collections import defaultdict


def stratified_premise_sample(data_path, output_path, samples_per_difficulty=402, random_seed=42):
    """
    Random selection of instances with uniform distribution across difficulty scores (402 for all [0, 0.2, 0.5, 0.75, 1] for divisibility)
    """
    random.seed(random_seed)
    print("Reading data...")

    # Read data more efficiently by only loading needed columns
    df = pd.read_csv(data_path, usecols=['unique_id', 'difficulty_score'])

    # Pre-compute all difficulty scores and initialize counts
    all_difficulties = sorted(df['difficulty_score'].unique())
    difficulty_counts = {diff: 0 for diff in all_difficulties}
    print(f"Found difficulty scores: {all_difficulties}")

    # More efficient premise key extraction
    df['premise_key'] = df['unique_id'].str.extract(r'(\d+_\d+)')[0]

    # Create efficient lookup dictionaries
    print("Creating lookup dictionaries...")
    difficulty_to_premises = defaultdict(set)
    premises_difficulties = {}
    premise_to_rows = {}

    # Single pass through data to build all lookups
    for premise_key, group in df.groupby('premise_key'):
        difficulties = set(group['difficulty_score'])
        premises_difficulties[premise_key] = difficulties
        premise_to_rows[premise_key] = group.index.tolist()
        for diff in difficulties:
            difficulty_to_premises[diff].add(premise_key)

    selected_premises = set()
    selected_indices = []

    print("Starting sampling process...")
    iterations = 0
    max_iterations = len(df) * 2  # Safety limit

    while min(difficulty_counts.values()) < samples_per_difficulty and iterations < max_iterations:
        iterations += 1
        if iterations % 100 == 0:
            print(f"Iteration {iterations}, current counts: {difficulty_counts}")

        # Find difficulty score needing most samples
        target_difficulty = min(difficulty_counts.items(), key=lambda x: x[1])[0]

        # Get available premises more efficiently
        available_premises = difficulty_to_premises[target_difficulty] - selected_premises
        if not available_premises:
            continue

        # Select a random premise
        selected_premise = random.choice(list(available_premises))

        # Check if selection would exceed limits
        valid = True
        temp_counts = difficulty_counts.copy()
        group_indices = premise_to_rows[selected_premise]
        for idx in group_indices:
            diff = df.iloc[idx]['difficulty_score']
            temp_counts[diff] += 1
            if temp_counts[diff] > samples_per_difficulty:
                valid = False
                break

        if valid:
            selected_premises.add(selected_premise)
            selected_indices.extend(group_indices)
            difficulty_counts = temp_counts

    print("\nFinal difficulty score distribution:")
    selected_df = df.iloc[selected_indices]
    final_counts = selected_df['difficulty_score'].value_counts()
    print(final_counts)

    # Save results with both columns
    print(f"\nSaving selected data to {output_path}")
    selected_df[['unique_id', 'difficulty_score']].to_csv(output_path, index=False)

    print(f"Sampling completed in {iterations} iterations")
    return selected_df[['unique_id', 'difficulty_score']]

# Example usage:
sampled_data = stratified_premise_sample(
     'results/combined_difficulty_scores.csv',
     'results/annealing_diff_preserved.csv'
 )
