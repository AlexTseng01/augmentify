Augmentify is a CLI data augmentation tool used to assist in YOLO model training.



#### **How to install (Windows 11)**



1. Install Python from '*https://www.python.org/downloads/*' (I recommend version 3.11.8)
2. Open a shell session
3. cd into your preferred directory to install the program
4. Once in the preferred directory, run: '*git clone https://github.com/AlexTseng01/augmentify.git*'
5. Create a virtual environment: '*python -m venv venv*'
6. Enter the virtual environment: '*./venv/Scripts/Activate.ps1'*
7. Install the dependencies: '*pip install albumentations opencv-python*'



Before using Augmentify, make sure you are in the Augmentify directory and that the virtual environment is activated. Then you can run the program.



#### **How to configure commands**



Once you have Augmentify installed and your virtual environment activated in the terminal, you can configure the program in several ways.



Augmentify requires two mandatory parameters to run; the target folder path that you want to augment, and some action(s) to perform on the entire target folder.



The target folder can be any folder with your dataset as long as Augmentify has permission to access it.



The action(s) can be any of these functions listed:

* h\_flip
* v\_flip
* rotate
* scale
* shift
* brightness
* contrast
* saturation
* grayscale
* invert
* add\_empty\_labels



**Optional Parameters**

| Parameter           | Type      | Description                                        | Example               |

|---------------------|-----------|----------------------------------------------------|-----------------------|

| `--save\_path`       | string    | Directory where output images will be saved.       | `"path/to/folder"`    |

| `--include\_path`    | boolean   | Include the original directory path in metadata.   | `True`                |

| `--rot\_deg`         | float     | Rotation (in degrees) to apply.                    | `-10.5`               |

| `--bright\_mult`     | float     | Brightness multiplier.                             | `2.0`                 |

| `--contrast\_mult`   | float     | Contrast multiplier.                               | `1.1`                 |



**Examples of configurations:**

1. Vertical flip + horizontal flip + brightness + save path + subdirectory

```bash

python augmentify\_cli.py "C:\\path\\to\\target" v\_flip h\_flip brightness --save\_path "C:\\path\\to\\saveFolder" --bright\_mult 2.0 --include\_sub True

```



2\. Horizontal flip only

```bash

*python augmentify\_cli.py "C:\\path\\to\\target" h\_flip*

```



3\. Add empty labels

```bash

*python augmentify\_cli.py "C:\\path\\to\\target" add\_empty\_labels*

```

