# CLI Example (Full): 
# python driver.py 
# "C:\path\to\target" --save_path "C:\path\to\saveFGlder" --include_sub True 
# --replace_files False --preview_img "C:\path\to\previewImg.png"

# CLI Example 2 (Partial):
# python driver.py C:\path\to\target --include_sub True --replace_files True

import argparse
import albumentations as album
import os

TARGET_PATH = None
SAVE_PATH = None
INCLUDE_SUB = False
REPLACE_FILES = False
PREVIEW_IMG = None

def flip_img():
    None

def filter_img():
    None

def rotate_img():
    None

def scale_img():
    None

def shift_img():
    None

def pad_img():
    None

# Gives all image files a corresponding empty label file, 
# this is mostly useful when training YOLO on negative datasets
def add_empty_labels():
    all_files = os.listdir(TARGET_PATH)
    existing_txts = {os.path.splitext(f)[0] for f in all_files if f.lower().endswith(".txt")}

    print()
    for item in all_files:
        path = os.path.join(TARGET_PATH, item)
        # Needs additional control structure for if the user wants to process all subdirectories
        if os.path.isfile(path) and item.lower().endswith((".png", ".jpg", ".jpeg")):
            img_name = os.path.splitext(item)[0]
            if img_name not in existing_txts:
                txt_path = os.path.join(TARGET_PATH, img_name + ".txt")
                with open(txt_path, "w") as f:
                    pass
                print(f"Created empty label for {item}")

# Helper function, ignore
def str_to_bool(s):
    if isinstance(s, bool):
        return s
    return s.lower() in ("true", "1", "yes")

def main(target, save, include_sub, replace_files, preview_img):
    global TARGET_PATH, SAVE_PATH, INCLUDE_SUB, REPLACE_FILES, PREVIEW_IMG
    TARGET_PATH = target
    SAVE_PATH = save
    INCLUDE_SUB = include_sub
    REPLACE_FILES = replace_files
    PREVIEW_IMG = preview_img

    print("TARGET_PATH =",TARGET_PATH)
    print("SAVE_PATH =", SAVE_PATH)
    print("INCLUDE_SUB =", INCLUDE_SUB)
    print("REPLACE_FILES =", REPLACE_FILES)
    print("PREVIEW_IMG =", PREVIEW_IMG)
    add_empty_labels()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augmentify")
    parser.add_argument("target_path", help="Path to target folder")
    parser.add_argument("--save_path", default=None, help="Path to save folder")
    parser.add_argument("--include_sub", type=str_to_bool, default=False, help="Include subfolders?")
    parser.add_argument("--replace_files", type=str_to_bool, default=False, help="Replace files?")
    parser.add_argument("--preview_img", default=None, help="Path to preview image")
    args = parser.parse_args()
    main(args.target_path, args.save_path, args.include_sub, args.replace_files, args.preview_img)

