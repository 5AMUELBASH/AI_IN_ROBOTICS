import os
import shutil
from pathlib import Path

def copy_subset_images(source_dir, dest_dir, num_files=2200, file_extensions=('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    dest_path.mkdir(parents=True, exist_ok=True)

    subdirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(subdirs)} folders in source directory")
    
    for subdir in subdirs:
        print(f"Processing folder: {subdir.name}")

        dest_subdir = dest_path / subdir.name
        dest_subdir.mkdir(exist_ok=True)

        image_files = []
        for ext in file_extensions:
            image_files.extend(subdir.glob(f'*{ext}'))
            image_files.extend(subdir.glob(f'*{ext.upper()}'))

        image_files = sorted(list(set(image_files)))
        
        print(f"  Found {len(image_files)} image files")

        files_to_copy = image_files[:num_files]

        copied_count = 0
        for file_path in files_to_copy:
            try:
                shutil.copy2(file_path, dest_subdir / file_path.name)
                copied_count += 1
            except Exception as e:
                print(f"  Error copying {file_path}: {e}")
        
        print(f"  Copied {copied_count} files to {dest_subdir.name}")
    
    print("Copy operation completed!")

if __name__ == "__main__":
    source_directory = "..."
    destination_directory = "..."
    
    copy_subset_images(source_directory, destination_directory, num_files=2200)