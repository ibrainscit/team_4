import os
import shutil

# List of folder names and their corresponding text files
folder_files = {
    '../other_files/mdd': 'm.txt'
}

# Create the folders if they don't already exist
for folder in folder_files:
    if not os.path.exists(folder):
        os.mkdir(folder)

# Move files based on their corresponding text file
for folder, txt_file in folder_files.items():
    # Check if the text file exists
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as file:
            file_names = file.readlines()

        # Move each file listed in the text file
        for file_name in file_names:
            file_name = file_name.strip()  # Remove any extra whitespace or newline
            if os.path.exists(file_name):
                shutil.move(file_name, folder)
            else:
                print(f"File '{file_name}' not found, skipping.")