import os
import re

# Directory containing the images
directory = "C:\\Users\\farsh\\Documents\\Mini\\111"
pad_length = 4  # Adjust the padding length based on the number of images

# Function to extract number from filename
def extract_number(filename):
    match = re.search(r'_(\d+)\.png$', filename)
    return int(match.group(1)) if match else 0

# Get a list of all image files and sort by the extracted number
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.png')]
files.sort(key=extract_number)

# Print the list of files to verify the initial order
print("Files before renaming:")
for file in files:
    print(file)

# Rename files with zero-padding
for count, filename in enumerate(files, start=1):
    new_name = f"{str(count).zfill(pad_length)}.png"
    old_file_path = os.path.join(directory, filename)
    new_file_path = os.path.join(directory, new_name)
    
    print(f"Renaming {old_file_path} to {new_file_path}")
    
    try:
        os.rename(old_file_path, new_file_path)
    except Exception as e:
        print(f"Error renaming file {old_file_path} to {new_file_path}: {e}")

print("Renaming complete!")

# List all files in the directory and print their names to verify the order after renaming
new_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.png')]
new_files.sort(key=extract_number)

print("Files after renaming:")
for file in new_files:
    print(file)
