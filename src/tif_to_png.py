# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:37:39 2024

@author: farsh
"""

from PIL import Image
import os

# Function to convert TIFF to PNG
def convert_tiff_to_png(source_folder, target_folder):
    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Walk through the source folder
    for subdir, dirs, files in os.walk(source_folder):
        for file in files:
            # Check if the file is a TIFF
            if file.lower().endswith(".tif") or file.lower().endswith(".tiff"):
                # Form the full file paths
                source_file = os.path.join(subdir, file)
                target_file = os.path.join(target_folder, os.path.relpath(subdir, source_folder), file.rsplit('.', 1)[0] + ".png")

                # Create subdirectory in target folder if it doesn't exist
                target_subdir = os.path.dirname(target_file)
                if not os.path.exists(target_subdir):
                    os.makedirs(target_subdir)

                # Open the image and convert it to PNG
                with Image.open(source_file) as img:
                    img.save(target_file, "PNG")

                #print(f"Converted and saved: {target_file}")

# Create a sample TIFF image
source_folder = 'C:\\Users\\farsh\\Documents\Work\\Test data\\crop\\DESY2022-Q4_HV150_7075_d20_002_TD8001_v150_p600_w0_Sw'
target_folder = 'C:\\Users\\farsh\\Documents\Work\\Test data\\crop\\DESY2022-Q4_HV150_7075_d20_002_TD8001_v150_p600_w0_Sw\\PNG'
os.makedirs(source_folder, exist_ok=True)

# Create a simple image
#img = Image.new('RGB', (100, 100), color = 'red')
#img.save(os.path.join(source_folder, 'test_image.tif'))

# Now let's use the function to convert this TIFF to PNG
convert_tiff_to_png(source_folder, target_folder)