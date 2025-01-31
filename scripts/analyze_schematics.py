import os

import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for the progress bar


def list_subdirectories_with_file_counts(directory):
    subdir_file_counts = {}

    # First count total directories
    total_dirs = sum(len(dirs) for _, dirs, _ in os.walk(directory))

    # Create progress bar with total count
    with tqdm(total=total_dirs, desc="Processing directories") as pbar:
        for root, dirs, files in os.walk(directory):
            for dir_name in dirs:
                subdir_path = os.path.join(root, dir_name)
                # Only count .schem files
                file_count = sum(
                    1
                    for entry in os.scandir(subdir_path)
                    if entry.is_file() and entry.name.endswith(".schem")
                )
                if file_count > 0:
                    # Get relative path from the root directory
                    rel_path = os.path.relpath(subdir_path, directory)
                    subdir_file_counts[rel_path] = file_count
                pbar.update(1)

    return subdir_file_counts


def plot_subdirectory_file_counts(subdir_file_counts):
    # Set the background for dark mode
    plt.style.use("dark_background")

    # Sort subdirectories by the number of files (descending)
    sorted_subdirs = sorted(
        subdir_file_counts.items(), key=lambda item: item[1], reverse=True
    )[:50]  # Only take top 50

    # Prepare data for plotting
    directories = [item[0] for item in sorted_subdirs]
    counts = [item[1] for item in sorted_subdirs]

    # Create a bar chart
    plt.figure(figsize=(12, 8))
    plt.barh(directories, counts, color="skyblue")
    plt.xlabel("Number of Files")
    plt.ylabel("Subdirectories")
    plt.title("Top 50 Subdirectories by Number of Files")
    plt.gca().invert_yaxis()

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.4)
    plt.show()


def main():
    directory = "data/schematics"
    subdir_file_counts = list_subdirectories_with_file_counts(directory)
    plot_subdirectory_file_counts(subdir_file_counts)


if __name__ == "__main__":
    main()
