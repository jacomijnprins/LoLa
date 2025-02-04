import os
import copy
import pandas as pd
import torch
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_checkpoint = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)


class CurriculumDataset(Dataset):
    def __init__(self, dataframe):
        # Drop any na rows
        dataframe = dataframe.dropna(subset=['premise', 'hypothesis'])
        dataframe = dataframe[dataframe['true_label'] != -1]

        # Convert labels to 0,1,2
        self.labels = dataframe['true_label'].astype(int).tolist()

        # Tokenize premise & hypothesis for RoBERTa
        self.encodings = tokenizer(
            list(dataframe['premise']),
            list(dataframe['hypothesis']),
            truncation=True,
            padding=True,
            max_length=128
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def train_annealing_curriculum(
    csv_file = "results/annealing_diff_preserved.csv",
    n_buckets = 5,      
    final_epochs = 2, 
    batch_size = 16
):
    """
    Implements Xu et al.'s multi-stage curriculum:
      - Splits the data into N buckets from easiest (difficulty score = 1) to hardest (difficulty_score = 0).
      - Train stage by stage, each stage i adding the next bucket.
      - Train for 2 epochs (N+1) on the full dataset.
    """

    print(f"Loading sorted data from {csv_file}")
    df = pd.read_csv(csv_file)

    df = df.sort_values(by="difficulty_score", ascending=True).reset_index(drop=True)

    # 1) Split the dataset into N buckets
    total_len = len(df)
    bucket_size = total_len // n_buckets
    buckets = []
    start_idx = 0
    for i in range(n_buckets):
        # Last bucket takes all leftover rows
        end_idx = start_idx + bucket_size if i < n_buckets - 1 else total_len
        bucket_df = df.iloc[start_idx:end_idx].copy()
        buckets.append(bucket_df)
        start_idx = end_idx
      
    base_dir = "models/curriculum_runs"
    os.makedirs(base_dir, exist_ok=True)

    # Prepare model & base training args
    model = RobertaForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3).to(device)
    base_training_args = TrainingArguments(
        output_dir=os.path.join(base_dir, "base_output"), 
        evaluation_strategy='no',
        save_strategy='no', 
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        logging_dir=os.path.join(base_dir, "logs"),
        logging_steps=10,
        lr_scheduler_type='linear',
        load_best_model_at_end=False
    )

    for stage_idx in range(n_buckets):
        stage_df = pd.concat(buckets[:stage_idx + 1], ignore_index=True)

        # Build a dataset for buckets
        stage_dataset = CurriculumDataset(stage_df)

        stage_training_args = copy.deepcopy(base_training_args)
        stage_training_args.num_train_epochs = 1
        stage_training_args.output_dir = os.path.join(base_dir, f"stage_{stage_idx+1}")

        trainer = Trainer(
            model=model,
            args=stage_training_args,
            train_dataset=stage_dataset,
            tokenizer=tokenizer
        )

        print(f"\n=== Stage {stage_idx + 1}/{n_buckets} ===")
        print(f"  -> Using buckets [0..{stage_idx}] => {len(stage_df)} examples")
        trainer.train()
        # (model is updated in-place)

    # Finally, stage N+1 => train on the FULL data for 2 more epochs
    full_dataset = CurriculumDataset(df)
    final_training_args = copy.deepcopy(base_training_args)
    final_training_args.num_train_epochs = final_epochs
    final_training_args.output_dir = os.path.join(base_dir, "stage_final")

    trainer = Trainer(
        model=model,
        args=final_training_args,
        train_dataset=full_dataset,
        tokenizer=tokenizer
    )
    print(f"\n=== Stage {n_buckets + 1}: Full dataset for {final_epochs} epoch(s) ===")
    trainer.train()

    final_save_dir = os.path.join(base_dir, "final_model")
    trainer.save_model(final_save_dir)
    print(f"\nAll stages complete. Final model saved to {final_save_dir}")

if __name__ == "__main__":
    train_annealing_curriculum(
        csv_file="results/annealing_diff_preserved.csv",
        n_buckets=5,   
        final_epochs=2,  
        batch_size=16
    )
