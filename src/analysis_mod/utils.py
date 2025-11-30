import os
import re


def get_latest_folder(base_directory: str) -> str:
    folders = [f for f in os.listdir(base_directory)
               if os.path.isdir(os.path.join(base_directory, f))]

    if not folders:
        raise ValueError(f"No folders found in {base_directory}")

    folders.sort(reverse=True)

    return os.path.join(base_directory, folders[0])


def latest_saved_individual_file(directory: str) -> str:
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Extract numeric parts from filenames matching pattern like "generation_0000.pkl"
    pattern = re.compile(r'generation_(\d+)\.pkl')
    max_num = -1
    latest_file_name = None

    for filename in files:
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                latest_file_name = filename

    if latest_file_name:
        return os.path.join(directory, latest_file_name)
    else:
        raise FileNotFoundError("No generation files found in directory")
