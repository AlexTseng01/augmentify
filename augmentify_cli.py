# Important Notes:
# 1. When using the CLI version, two parameters MUST be present; the target directory, and the action you want to perform.
# 2. Every other parameter is optional and can be omitted entirely. Refer to {CLI Example 3}.
# 3. If you want to include all parameters, refer to {CLI Example 1}.

# CLI Example 1 (Full): 
# python augmentify_cli.py "C:\path\to\target" action --save_path "C:\path\to\saveFolder" --include_sub True

# CLI Example 2 (Partial):
# python augmentify_cli.py "C:\path\to\target" action --include_sub True

# CLI Example 3 (Minimum):
# python augmentify_cli.py "C:\path\to\target" action

import argparse
import albumentations as A
import os
import cv2
import sys
import shutil
import numpy as np

TARGET_PATH = None
SAVE_PATH = None
INCLUDE_SUB = False

# Flips image horizontally
def h_flip(file_path, existing_txts):
    transform = A.Compose([A.HorizontalFlip(p=1)])
    dirpath = os.path.dirname(file_path)
    file = os.path.basename(file_path)
    img_name = os.path.splitext(file)[0]

    print(f"Applying horizontal flip to {os.path.join(dirpath, file)}\n")

    # Flip image
    image = cv2.imread(file_path)
    flipped_img = transform(image=image)["image"]
    output_image_path = os.path.join(SAVE_PATH, img_name + ".png")
    cv2.imwrite(output_image_path, flipped_img)

    # Flip corresponding label if it exists
    if img_name in existing_txts:
        label_path = os.path.join(dirpath, img_name + ".txt")

        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            cls, x_center, y_center, w, h = line.strip().split()
            x_center = 1 - float(x_center)  # horizontal flip
            y_center = float(y_center)
            w = float(w)
            h = float(h)
            new_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        # Writing to SAVE_PATH
        output_label_path = os.path.join(SAVE_PATH, img_name + ".txt")
        with open(output_label_path, "w") as f:
            f.writelines(new_lines)

# Flips image vertically
def v_flip(file_path, existing_txts):
    transform = A.Compose([A.VerticalFlip(p=1)])
    dirpath = os.path.dirname(file_path)
    file = os.path.basename(file_path)
    img_name = os.path.splitext(file)[0]

    print(f"Applying vertical flip to {os.path.join(dirpath, file)}\n")

    # Flip image
    image = cv2.imread(file_path)
    flipped_img = transform(image=image)["image"]
    output_image_path = os.path.join(SAVE_PATH, img_name + ".png")
    cv2.imwrite(output_image_path, flipped_img)

    # Flip corresponding label if it exists
    if img_name in existing_txts:
        label_path = os.path.join(dirpath, img_name + ".txt")

        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            cls, x_center, y_center, w, h = line.strip().split()
            x_center = float(x_center)        # horizontal position stays the same
            y_center = 1 - float(y_center)    # vertical flip
            w = float(w)
            h = float(h)
            new_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        # Writing to SAVE_PATH
        output_label_path = os.path.join(SAVE_PATH, img_name + ".txt")
        with open(output_label_path, "w") as f:
            f.writelines(new_lines)

# Rotate image by some degree value
def rotate():
    None

# Zoom image by some multiplier value
def scale():
    None

# Shifts by some x-axis or y-axis value
def shift():
    None

# Adjust brightness values
def brightness(file_path, existing_txts):
    dirpath = os.path.dirname(file_path)
    file = os.path.basename(file_path)
    img_name = os.path.splitext(file)[0]

    print(f"Applying brightness settings to {os.path.join(dirpath, file)}\n")

    # Editing images
    img = cv2.imread(file_path)
    bright_img = cv2.convertScaleAbs(img, alpha=1.0, beta=50)
    output_image_path = os.path.join(SAVE_PATH, file)
    cv2.imwrite(output_image_path, bright_img)

    # Copying labels
    label_path = os.path.join(dirpath, img_name + ".txt")
    output_label_path = os.path.join(SAVE_PATH, img_name + ".txt")

    # Writing to SAVE_PATH
    if os.path.exists(output_label_path):
        return

    original_label_path = os.path.join(os.path.dirname(file_path), img_name + ".txt")
    if os.path.exists(original_label_path):
        shutil.copy(original_label_path, output_label_path)

# Adjust contrast values
def contrast(file_path, existing_txts):
    dirpath = os.path.dirname(file_path)
    file = os.path.basename(file_path)
    img_name = os.path.splitext(file)[0]
    
    print(f"Applying contrast settings to {os.path.join(TARGET_PATH, file)}\n")

    # Editing images
    img = cv2.imread(os.path.join(dirpath, file))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation_factor = 1.5
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)
    hsv_adjusted = cv2.merge([h, s, v])

    saturated_img = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    output_image_path = os.path.join(SAVE_PATH, file)
    cv2.imwrite(output_image_path, saturated_img)

    # Copying labels
    label_path = os.path.join(TARGET_PATH, img_name + ".txt")
    output_label_path = os.path.join(SAVE_PATH, img_name + ".txt")
    
    # Writing to SAVE_PATH
    if os.path.exists(output_label_path):
        return

    original_label_path = os.path.join(os.path.dirname(file_path), img_name + ".txt")
    if os.path.exists(original_label_path):
        shutil.copy(original_label_path, output_label_path)

# Adjust saturation values
def saturation(file_path, existing_txts):
    dirpath = os.path.dirname(file_path)
    file = os.path.basename(file_path)
    img_name = os.path.splitext(file)[0]

    print(f"Applying saturation settings to {os.path.join(dirpath, file)}\n")

    # Editing images
    img = cv2.imread(os.path.join(dirpath, file))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation_factor = 1.5
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)
    hsv_adjusted = cv2.merge([h, s, v])

    saturated_img = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    output_image_path = os.path.join(SAVE_PATH, file)
    cv2.imwrite(output_image_path, saturated_img)

    # Copying labels
    label_path = os.path.join(dirpath, img_name + ".txt")
    output_label_path = os.path.join(SAVE_PATH, img_name + ".txt")

    # Writing to SAVE_PATH
    if os.path.exists(output_label_path):
        return

    original_label_path = os.path.join(os.path.dirname(file_path), img_name + ".txt")
    if os.path.exists(original_label_path):
        shutil.copy(original_label_path, output_label_path)

# Turns the image black and white
def grayscale():
    None

# Invert colors of the image
def invert():
    None

# Gives all image files a corresponding empty label file, 
def add_empty_labels(file_path, existing_txts):
    dirpath = os.path.dirname(file_path)
    file = os.path.basename(file_path)
    img_name = os.path.splitext(file)[0]

    if img_name not in existing_txts:
        # Creating empty labels
        label_path = os.path.join(dirpath, img_name + ".txt")
        with open (label_path, "w") as f:
            pass
        
        file_path = os.path.join(dirpath, file)
        print(f"Created label file for: {file_path}\n")
                            
# Helper functions
def str_to_bool(s):
    if isinstance(s, bool):
        return s
    return s.lower() in ("true", "1", "yes")

# Helper function that processes the necessary file information before calling the desired action
def dir_traverse(action):
    if INCLUDE_SUB:
        for dirpath, dirname, filenames in os.walk(TARGET_PATH):
            existing_txts = {os.path.splitext(f)[0] for f in filenames if f.lower().endswith(".txt")}
            for file in filenames:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(dirpath, file)
                    print(f"Processing: {full_path}")

                    # Call the specific action
                    action(full_path, existing_txts)
    else:
        all_files = os.listdir(TARGET_PATH)
        existing_txts = {os.path.splitext(f)[0] for f in all_files if f.lower().endswith(".txt")}
        for file in all_files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(TARGET_PATH, file)
                print(f"Processing: {full_path}")

                # Call the specific action
                action(full_path, existing_txts)

def main(target, actions, save, include_sub):
    global TARGET_PATH, SAVE_PATH, INCLUDE_SUB
    TARGET_PATH = target
    SAVE_PATH = save
    INCLUDE_SUB = include_sub

    # If SAVE_PATH does not exist, create a new folder inside TARGET_PATH, otherwise, save it into SAVE_PATH
    if not SAVE_PATH or SAVE_PATH.strip() == "" or not os.path.exists(SAVE_PATH):
        SAVE_PATH = os.path.join(TARGET_PATH, "augmented_images")
    os.makedirs(SAVE_PATH, exist_ok=True)

    # List of actions the user can perform in sequence
    ACTION_MAP = {
        "h_flip": h_flip,
        "v_flip": v_flip,
        "rotate": rotate,
        "scale": scale,
        "shift": shift,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "grayscale": grayscale,
        "invert": invert,
        "add_empty_labels": add_empty_labels
    }

    # Args validation
    for action in actions:
        if action not in ACTION_MAP:
            print(f"Error: '{action}' is not an action.")
            sys.exit(1)
    
    # Exec actions
    for action in actions:
        print("---------------------------------------------------------------------------------------------------")
        dir_traverse(ACTION_MAP[action])
        TARGET_PATH = SAVE_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augmentify")
    parser.add_argument("target_path", help="Path to target folder")
    parser.add_argument("action", nargs="*", help="Action(s) to perform")
    parser.add_argument("--save_path", default=None, help="Path to save folder")
    parser.add_argument("--include_sub", type=str_to_bool, default=False, help="Include subdirectories")
    args = parser.parse_args()
    main(args.target_path, args.action, args.save_path, args.include_sub)
