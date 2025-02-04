import os
import pandas as pd


def combine_all(output_file, metaset_count=5):

    with open(output_file, 'w') as out:
        out.write("unique_id,difficulty_score\n")

        for metaset_idx in range(metaset_count):
            file_path = f"cross_review/results/difficulty_scores_metaset_{metaset_idx}.csv"

            if not os.path.exists(file_path):
                print(f"file {file_path} does not exist")
                continue

            print(f"processing {file_path}")

            data = pd.read_csv(file_path)

            if 'unique_id' in data.columns and 'difficulty_score' in data.columns:
                data[['unique_id', 'difficulty_score']].to_csv(out, index=False, header=False, mode='a')
            else:
                print(f"missing required cols in {file_path}")

    combined_data = pd.read_csv(output_file)
    cleaned_data = combined_data.dropna()
    cleaned_data.to_csv(output_file, index=False)
    print(f"combined file saved to {output_file}")

output_file = "results/combined_difficulty_scores.csv"

combine_all(output_file)
