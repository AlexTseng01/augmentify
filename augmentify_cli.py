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

TARGET_PATH = None
SAVE_PATH = None
INCLUDE_SUB = False

def h_flip():
    global SAVE_PATH

    transform = A.Compose([A.HorizontalFlip(p=1)])

    if not SAVE_PATH or SAVE_PATH.strip() == "" or not os.path.exists(SAVE_PATH):
        SAVE_PATH = os.path.join(TARGET_PATH, "augmented_images")
    os.makedirs(SAVE_PATH, exist_ok=True)

    if INCLUDE_SUB:
        for dirpath, dirname, filenames in os.walk(TARGET_PATH):
            existing_txts = {os.path.splitext(f)[0] for f in filenames if f.lower().endswith(".txt")}
            for file in filenames:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_name = os.path.splitext(file)[0]

                    # Flip image
                    file_path = os.path.join(dirpath, file)
                    image = cv2.imread(file_path)
                    flipped_img = transform(image=image)["image"]
                    output_image_path = os.path.join(SAVE_PATH, img_name + "_h_flip.png")
                    cv2.imwrite(output_image_path, flipped_img)
                    print(f"Flipped image saved to {output_image_path}")

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

                        output_label_path = os.path.join(SAVE_PATH, img_name + "_h_flip.txt")
                        with open(output_label_path, "w") as f:
                            f.writelines(new_lines)
                        print(f"Flipped label saved to {output_label_path}")
    else:
        all_files = os.listdir(TARGET_PATH)
        existing_txts = {os.path.splitext(f)[0] for f in all_files if f.lower().endswith(".txt")}
        for file in os.listdir(TARGET_PATH):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_name = os.path.splitext(file)[0]

                # Flip image
                file_path = os.path.join(TARGET_PATH, file)
                image = cv2.imread(file_path)
                flipped_img = transform(image=image)["image"]
                output_image_path = os.path.join(SAVE_PATH, img_name + "_h_flip.png")
                cv2.imwrite(output_image_path, flipped_img)
                print(f"Horizontally flipped image saved to {output_image_path}")

                # Flip corresponding label if it exists
                if img_name in existing_txts:
                    label_path = os.path.join(TARGET_PATH, img_name + ".txt")
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

                    output_label_path = os.path.join(SAVE_PATH, img_name + "_h_flip.txt")
                    with open(output_label_path, "w") as f:
                        f.writelines(new_lines)
                    print(f"Horizontally flipped label saved to {output_label_path}")

def v_flip():
    global SAVE_PATH

    transform = A.Compose([A.VerticalFlip(p=1)])

    if not SAVE_PATH or SAVE_PATH.strip() == "" or not os.path.exists(SAVE_PATH):
        SAVE_PATH = os.path.join(TARGET_PATH, "augmented_images")
    os.makedirs(SAVE_PATH, exist_ok=True)

    if INCLUDE_SUB:
        for dirpath, dirname, filenames in os.walk(TARGET_PATH):
            existing_txts = {os.path.splitext(f)[0] for f in filenames if f.lower().endswith(".txt")}
            for file in filenames:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_name = os.path.splitext(file)[0]

                    # Flip image
                    file_path = os.path.join(dirpath, file)
                    image = cv2.imread(file_path)
                    flipped_img = transform(image=image)["image"]
                    output_image_path = os.path.join(SAVE_PATH, img_name + "_v_flip.png")
                    cv2.imwrite(output_image_path, flipped_img)
                    print(f"Vertically flipped image saved to {output_image_path}")

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
                        
                        output_label_path = os.path.join(SAVE_PATH, img_name + "_v_flip.txt")
                        with open(output_label_path, "w") as f:
                            f.writelines(new_lines)
                        print(f"Vertically flipped label saved to {output_label_path}")
    else:
        all_files = os.listdir(TARGET_PATH)
        existing_txts = {os.path.splitext(f)[0] for f in all_files if f.lower().endswith(".txt")}
        for file in os.listdir(TARGET_PATH):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_name = os.path.splitext(file)[0]

                # Flip image
                file_path = os.path.join(TARGET_PATH, file)
                image = cv2.imread(file_path)
                flipped_img = transform(image=image)["image"]
                output_image_path = os.path.join(SAVE_PATH, img_name + "_v_flip.png")
                cv2.imwrite(output_image_path, flipped_img)
                print(f"Flipped image saved to {os.path.join(output_image_path)}")

                # Flip corresponding label if it exists
                if img_name in existing_txts:
                    label_path = os.path.join(TARGET_PATH, img_name + ".txt")
                    with open(label_path, "r") as f:
                        lines = f.readlines()

                    new_lines = []
                    for line in lines:
                        cls, x_center, y_center, w, h = line.strip().split()
                        x_center = float(x_center)        # horizontal stays the same
                        y_center = 1 - float(y_center)    # vertical flip
                        w = float(w)
                        h = float(h)
                        new_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

                    output_label_path = os.path.join(SAVE_PATH, img_name + "_v_flip.txt")
                    with open(output_label_path, "w") as f:
                        f.writelines(new_lines)
                    print(f"Flipped label saved to {output_label_path}")

def rotate():
    None

def scale():
    None

def shift():
    None

# Gives all image files a corresponding empty label file, 
def add_empty_labels():
    if INCLUDE_SUB:
        for dirpath, dirnames, filenames in os.walk(TARGET_PATH):
            existing_txts = {os.path.splitext(f)[0] for f in filenames if f.lower().endswith(".txt")}
            for file in filenames:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_name = os.path.splitext(file)[0]
                    if img_name not in existing_txts:
                        # Creating empty labels
                        label_path = os.path.join(dirpath, img_name + ".txt")
                        with open (label_path, "w") as f:
                            pass
                        
                        file_path = os.path.join(dirpath, file)
                        print(f"Created label file for: {file_path}") # Debug output
    else:
        all_files = os.listdir(TARGET_PATH)
        existing_txts = {os.path.splitext(f)[0] for f in all_files if f.lower().endswith(".txt")}
        for filename in os.listdir(TARGET_PATH):
            file_path = os.path.join(TARGET_PATH, filename)
            if os.path.isfile(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_name = os.path.splitext(filename)[0]
                if img_name not in existing_txts:
                    # Creating empty labels
                    label_path = os.path.join(TARGET_PATH, img_name + ".txt")
                    with open (label_path, "w") as f:
                        pass

                    print(f"Created label file for: {file_path}") # Debug output
                            
# Helper function, ignore
def str_to_bool(s):
    if isinstance(s, bool):
        return s
    return s.lower() in ("true", "1", "yes")

def main(target, action, save, include_sub):
    global TARGET_PATH, SAVE_PATH, INCLUDE_SUB, REPLACE_FILES
    TARGET_PATH = target
    SAVE_PATH = save
    INCLUDE_SUB = include_sub

    if action == "h_flip":
        h_flip()
    elif action == "v_flip":
        v_flip()
    elif action == "rotate":
        rotate()
    elif action == "scale":
        scale()
    elif action == "shift":
        shift()
    elif action == "add_empty_labels":
        add_empty_labels()
    else:
        print("Error: Invalid action call.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augmentify")
    parser.add_argument("target_path", help="Path to target folder")
    parser.add_argument("action", help="Specify action to perform")
    parser.add_argument("--save_path", default=None, help="Path to save folder")
    parser.add_argument("--include_sub", type=str_to_bool, default=False, help="Include subfolders?")
    parser.add_argument("--preview_img", default=None, help="Path to preview image")
    args = parser.parse_args()
    main(args.target_path, args.action, args.save_path, args.include_sub)

