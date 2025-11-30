# Important Notes:
# 1. When using the CLI version, two parameters MUST be present; the target directory, and the action you want to perform.
# 2. Every other parameter is optional and can be omitted entirely. Refer to {CLI Example 3}.
# 3. If you want to include all parameters, refer to {CLI Example 1}.

# CLI Example 1 (Full): 
# python augmentify_cli.py "C:\path\to\target" action1 action2 action3 action4 action5 --save_path "C:\path\to\saveFolder" --include_sub True

# CLI Example 2 (Partial):
# python augmentify_cli.py "C:\path\to\target" action1 action2 --include_sub True

# CLI Example 3 (Minimum requirement):
# python augmentify_cli.py "C:\path\to\target" action

# Known issue:
# When calling h_flip or v_flip together in any order, it causes corrupted labels

import argparse
import albumentations as A
import os
import cv2
import sys
import shutil
import numpy as np
import threading
import queue

TARGET_PATH = None
SAVE_PATH = None
INCLUDE_SUB = False

NUM_CONSUMERS = 4
queue_files = queue.Queue()
queue_results = queue.Queue()
stop_signal = object()

def producer_walk(target_path, include_sub):
    if include_sub:
        for dirpath, _, filenames in os.walk(target_path):
            for file in filenames:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    queue_files.put(os.path.join(dirpath, file))
    else:
        for file in os.listdir(target_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                queue_files.put(os.path.join(target_path, file))

    # signal consumers to stop
    for _ in range(NUM_CONSUMERS):
        queue_files.put(stop_signal)

def consumer_worker(actions, existing_txts):
    while True:
        file_path = queue_files.get()

        if file_path is stop_signal:
            queue_results.put(stop_signal)
            return

        img = cv2.imread(file_path)
        img_name = os.path.splitext(os.path.basename(file_path))[0]

        # apply actions in order
        for action in actions:
            img, label_data = action(img, img_name, existing_txts)

        queue_results.put((img_name, img, label_data))
        queue_files.task_done()

def writer_worker():
    stop_count = 0
    while True:
        item = queue_results.get()
        if item is stop_signal:
            stop_count += 1
            if stop_count == NUM_CONSUMERS:
                return
            continue
        img_name, img, label_data = item
        cv2.imwrite(os.path.join(SAVE_PATH, img_name + ".png"), img)
        if label_data is not None:
            with open(os.path.join(SAVE_PATH, img_name + ".txt"), "w") as f:
                f.writelines(label_data)
        queue_results.task_done()

# Flips image horizontally
def h_flip(img, img_name, existing_txts):
    flipped_img = cv2.flip(img, 1)
    label_data = None

    # Flip corresponding label if it exists
    if img_name in existing_txts:
        label_data = []
        label_path = os.path.join(TARGET_PATH, img_name + ".txt")
        with open(label_path, "r") as f:
            for line in f:
                cls, x, y, w, h = line.split()
                x = 1 - float(x) # horizontal flip
                y = float(y)
                w = float(w)
                h = float(h)
                label_data.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    return flipped_img, label_data

# Flips image vertically
def v_flip(img, img_name, existing_txts):
    flipped_img = cv2.flip(img, 0)
    label_data = None

    # Flip corresponding label if it exists
    if img_name in existing_txts:
        label_data = []
        label_path = os.path.join(TARGET_PATH, img_name + ".txt")
        with open(label_path, "r") as f:
            for line in f:
                cls, x, y, w, h = line.split()
                x = float(x)
                y = 1 - float(y) # vertical flip
                w = float(w)
                h = float(h)
                label_data.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    return flipped_img, label_data

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
def brightness(img, img_name, existing_txts):
    bright_img = cv2.convertScaleAbs(img, alpha=1.0, beta=50)
    label_data = None
    if img_name in existing_txts:
        label_path = os.path.join(TARGET_PATH, img_name + ".txt")
        with open(label_path, "r") as f:
            label_data = f.readlines()
    return bright_img, label_data

# Adjust contrast values
def contrast(img, img_name, existing_txts):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    hsv_adjusted = cv2.merge([h, s, v])
    contrast_img = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    label_data = None
    if img_name in existing_txts:
        label_path = os.path.join(TARGET_PATH, img_name + ".txt")
        with open(label_path, "r") as f:
            label_data = f.readlines()
    return contrast_img, label_data

# Adjust saturation values
def saturation(img, img_name, existing_txts):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    hsv_adjusted = cv2.merge([h, s, v])
    sat_img = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    label_data = None
    if img_name in existing_txts:
        label_path = os.path.join(TARGET_PATH, img_name + ".txt")
        with open(label_path, "r") as f:
            label_data = f.readlines()
    return sat_img, label_data

# Turns the image black and white
def grayscale(img, img_name, existing_txts):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    label_data = None
    if img_name in existing_txts:
        label_path = os.path.join(TARGET_PATH, img_name + ".txt")
        with open(label_path, "r") as f:
            label_data = f.readlines()
    return gray_img, label_data

# Invert colors of the image
def invert(img, img_name, existing_txts):
    inv_img = cv2.bitwise_not(img)
    label_data = None
    if img_name in existing_txts:
        label_path = os.path.join(TARGET_PATH, img_name + ".txt")
        with open(label_path, "r") as f:
            label_data = f.readlines()
    return inv_img, label_data

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
                            
# Helper function
def str_to_bool(s):
    if isinstance(s, bool):
        return s
    return s.lower() in ("true", "1", "yes")

# Creates the save directory with the target directory's/subdirectories files copied into it
def flat_copy():
    global TARGET_PATH, SAVE_PATH
    if not SAVE_PATH or SAVE_PATH.strip() == "" or not os.path.exists(SAVE_PATH):
        SAVE_PATH = os.path.join(TARGET_PATH, "augmented_images")
    os.makedirs(SAVE_PATH, exist_ok=True)

    if INCLUDE_SUB:
        for dirpath, dirnames, filenames in os.walk(TARGET_PATH):
            for file in filenames:
                s = os.path.join(dirpath, file)
                d = os.path.join(SAVE_PATH, file)
                shutil.copy2(s, d)
    else:
        s = os.path.join(TARGET_PATH, file)
        if os.path.isfile(s):
            shutil.copy2(s, SAVE_PATH)
    
    # Short-handed solution, delete later
    TARGET_PATH = SAVE_PATH

def main(target, actions, save, include_sub):
    global TARGET_PATH, SAVE_PATH, INCLUDE_SUB
    TARGET_PATH = target
    SAVE_PATH = save
    INCLUDE_SUB = include_sub

    flat_copy()

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

    for action_name in actions:
        action_func = ACTION_MAP[action_name]

        # Build list of existing labels
        all_files = os.listdir(TARGET_PATH)
        existing_txts = {os.path.splitext(f)[0] for f in all_files if f.lower().endswith(".txt")}

        # Start producer
        t_producer = threading.Thread(target=producer_walk, args=(TARGET_PATH, INCLUDE_SUB))
        t_producer.start()

        # Start consumers
        consumers = []
        for _ in range(NUM_CONSUMERS):
            t = threading.Thread(target=consumer_worker, args=([action_func], existing_txts))
            t.start()
            consumers.append(t)

        # Start writer (no arguments)
        t_writer = threading.Thread(target=writer_worker)
        t_writer.start()

        # Wait for threads to finish
        t_producer.join()
        for t in consumers:
            t.join()
        t_writer.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augmentify")
    parser.add_argument("target_path", help="Path to target folder")
    parser.add_argument("action", nargs="*", help="Action(s) to perform")
    parser.add_argument("--save_path", default=None, help="Path to save folder")
    parser.add_argument("--include_sub", type=str_to_bool, default=False, help="Include subdirectories")
    args = parser.parse_args()
    main(args.target_path, args.action, args.save_path, args.include_sub)
    print(f"Augmented dataset saved to: {SAVE_PATH}")