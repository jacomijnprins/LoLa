'''
Preprocessing Steps:
1. Take metasets as input
2. Adding unique_id columns to metaset to join metasets later after difficulty scores calculates
3. Added metaset index to ensure no unique_id clashes
'''

# Adding unique IDs to metaset files for later difficulty score matching
def preprocess_metasets(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each metaset file
    for metaset_file in os.listdir(input_folder):
        if metaset_file.endswith(".csv"):
            print(f"Processing {metaset_file}...")

            # Load metaset
            filepath = os.path.join(input_folder, metaset_file)
            df = pd.read_csv(filepath, header=None, names=["premise", "hypothesis", "label"])

            # Group by premise and filter premises with more than 3 hypotheses
            filtered_df = df.groupby("premise").filter(lambda x: len(x) == 3)

            # Assign unique IDs (e.g., 1a, 1b, 1c)
            unique_ids = []
            for idx, (premise, group) in enumerate(filtered_df.groupby("premise"), start=1):
                for sub_idx, (_, row) in enumerate(group.iterrows(), start=97):  # ASCII `a` = 97
                    unique_id = f"{idx}{chr(sub_idx)}"
                    unique_ids.append(unique_id)

            # Add IDs to the filtered dataframe
            filtered_df["unique_id"] = unique_ids

            # Save the processed metaset
            output_path = os.path.join(output_folder, metaset_file)
            filtered_df.to_csv(output_path, index=False)
            print(f"Saved preprocessed metaset to {output_path} with {len(filtered_df)} rows.")


# Input and output folder paths
input_folder = "data/processed"
output_folder = "data/processed"

preprocess_metasets(input_folder, output_folder)

def update_metaset_ids(metaset_count=5):
    for metaset_idx in range(metaset_count):
        file_path = f"data/processed/metaset_{metaset_idx}.csv"

        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping...")
            continue

        print(f"Processing {file_path}...")

        # Load the metaset file
        data = pd.read_csv(file_path)

        # Update unique_id to include the metaset index
        data['unique_id'] = data['unique_id'].apply(lambda x: f"{metaset_idx}_{x}")

        # Save the updated file back to the same location
        data.to_csv(file_path, index=False)
        print(f"Updated unique_id in {file_path}.")


# Update unique_ids in all metasets
update_metaset_ids(metaset_count=5)














