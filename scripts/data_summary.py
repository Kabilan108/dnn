import os
import pandas as pd

# Base directory containing the datasets
base_dir = "./data"  # replace with your directory path

# List of datasets
datasets = ["mri", "oct", "chest_xray"]

# List of splits
splits = ["train", "test", "val"]

# Initialize a list to hold the DataFrames
dfs = []

# Iterate over each dataset
for dataset in datasets:
    # Iterate over each split
    for split in splits:
        # Directory for this split
        split_dir = os.path.join(base_dir, dataset, split)

        # Get a list of all subdirectories in this split
        classes = [
            d
            for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ]

        # Iterate over each class
        for class_name in classes:
            # Directory for this class
            class_dir = os.path.join(split_dir, class_name)

            # Count the number of files in this class (excluding subdirectories)
            num_samples = len(
                [
                    f
                    for f in os.listdir(class_dir)
                    if os.path.isfile(os.path.join(class_dir, f))
                ]
            )

            # Add a DataFrame to the list
            dfs.append(
                pd.DataFrame(
                    {
                        "dataset": [dataset],
                        "split": [split],
                        "class": [class_name],
                        "samples": [num_samples],
                    }
                )
            )

# Concatenate all the DataFrames
df = pd.concat(dfs, ignore_index=True)

# Save the DataFrame to an Excel file
df.to_excel("dataset_summary.xlsx", index=False)
