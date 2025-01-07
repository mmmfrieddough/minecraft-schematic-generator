import os

import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for the progress bar


def list_subdirectories_with_file_counts(directory):
    subdir_file_counts = {}
    # Get all directories first to calculate progress
    all_dirs = [
        os.path.join(root, dir)
        for root, dirs, files in os.walk(directory)
        for dir in dirs
    ]
    for subdir_path in tqdm(all_dirs, desc="Processing directories"):
        file_count = len(
            [
                file
                for file in os.listdir(subdir_path)
                if os.path.isfile(os.path.join(subdir_path, file))
            ]
        )
        subdir_name = os.path.basename(subdir_path)
        subdir_file_counts[subdir_name] = file_count

    return subdir_file_counts


def plot_subdirectory_file_counts(subdir_file_counts):
    # Sort subdirectories by the number of files (descending)
    sorted_subdirs = sorted(
        subdir_file_counts.items(), key=lambda item: item[1], reverse=True
    )

    # Prepare data for plotting
    directories = [item[0] for item in sorted_subdirs]
    counts = [item[1] for item in sorted_subdirs]

    # Create a bar chart
    plt.figure(figsize=(12, 8))  # Increased width
    plt.barh(directories, counts, color="skyblue")
    plt.xlabel("Number of Files")
    plt.ylabel("Subdirectories")
    plt.title("Number of Files in Each Subdirectory")
    # Invert y-axis to have the directory with the most files on top
    plt.gca().invert_yaxis()

    plt.tight_layout()  # Adjust layout to make room
    # Adjust left margin to make more space for directory names
    plt.subplots_adjust(left=0.3)
    plt.show()


# Example usage
directory = input("Enter the path of the directory: ")
subdir_file_counts = list_subdirectories_with_file_counts(directory)
plot_subdirectory_file_counts(subdir_file_counts)
