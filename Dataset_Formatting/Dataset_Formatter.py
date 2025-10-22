import os
import shutil
import random
from pathlib import Path

def create_yolov8_classification_structure(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, but got {total_ratio}")

    if not source_path.exists():
        print(f"ERROR: Source directory '{source_path}' does not exist!")
        return

    class_folders = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not class_folders:
        print(f"No class folders found in '{source_path}'")
        return
    
    print(f"Found {len(class_folders)} classes: {[f.name for f in class_folders]}")

    random.seed(seed)

    splits = ['train', 'val', 'test']
    for split in splits:
        for class_folder in class_folders:
            (output_path / split / class_folder.name).mkdir(parents=True, exist_ok=True)

    for class_folder in class_folders:
        print(f"\nProcessing class: {class_folder.name}")

        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
        image_files = []
        for ext in image_extensions:
            image_files.extend(class_folder.glob(f'*{ext}'))
            image_files.extend(class_folder.glob(f'*{ext.upper()}'))

        image_files = list(set(image_files))
        random.shuffle(image_files)
        
        total_images = len(image_files)
        print(f"  Found {total_images} images")
        
        if total_images == 0:
            print(f"  WARNING: No images found in {class_folder.name}")
            continue

        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)

        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]

        split_counts = {}
        for split_name, split_files in zip(splits, [train_files, val_files, test_files]):
            count = 0
            for file_path in split_files:
                try:
                    dest_path = output_path / split_name / class_folder.name / file_path.name
                    shutil.copy2(file_path, dest_path)
                    count += 1
                except Exception as e:
                    print(f"    Error copying {file_path}: {e}")
            
            split_counts[split_name] = count
            print(f"  {split_name}: {count} images ({count/total_images*100:.1f}%)")
    
    print(f"\nDataset restructuring completed!")
    print(f"Output directory: {output_path}")
    print(f"Split ratios: Train {train_ratio*100}% | Val {val_ratio*100}% | Test {test_ratio*100}%")

def print_dataset_stats(dataset_path):
    """Print statistics about the created dataset"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print("Dataset path does not exist!")
        return
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if split_path.exists():
            class_folders = [d for d in split_path.iterdir() if d.is_dir()]
            total_images = 0
            print(f"\n{split.upper()} Split:")
            for class_folder in class_folders:
                image_count = len(list(class_folder.glob('*.*')))
                total_images += image_count
                print(f"  {class_folder.name}: {image_count} images")
            print(f"  Total: {total_images} images")

if __name__ == "__main__":
    SOURCE_DIRECTORY = "..."
    OUTPUT_DIRECTORY = "..."

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1

    create_yolov8_classification_structure(
        source_dir=SOURCE_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=42
    )

    print_dataset_stats(OUTPUT_DIRECTORY)