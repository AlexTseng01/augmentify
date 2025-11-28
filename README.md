Augmentify is a CLI data augmentation tool used to assist in YOLO model training. Features include: flipping images vertically/horizontally, image filters, rotation, scaling/zooming, translations/shifting, padding, empty labels for negative data, and more in future development.



#### **How to install (Windows 11)**



1. Install Python from '*https://www.python.org/downloads/*' (I recommend version 3.11.8)
2. Open a shell session
3. cd into your preferred directory to install the program
4. Once in the preferred directory, run: '*git clone https://github.com/AlexTseng01/augmentify.git*'
5. Create a virtual environment: '*python -m venv venv*'
6. Enter the virtual environment: '*./venv/Scripts/Activate.ps1'*
7. Install the dependencies: '*pip install albumentations opencv-python*'



You can now run Augmentify. Make sure to be in the Augmentify directory and enter the virtual environment every time you run this program.



#### **How to configure commands**



After you have your terminal session with Augmentify installed and entered your virtual environment, there are several ways you can configure it.

Augmentify requires two mandatory parameters to run; the target folder path that you want to augment, and one action to perform on the entire folder.



The target folder can be any folder as long as Augmentify has permission to access it.



The action can be any one of these listed:

* h\_flip
* v\_flip
* rotate
* scale
* shift
* add\_empty\_labels



There are two *optional* parameters you can include. It is *not* required and will run regardless of it being provided or not:

* save\_path accepts a path to a save folder
* include\_sub is a Boolean, accepting True or False



**Examples of configurations**:



***python augmentify\_cli.py "C:\\path\\to\\target" v\_flip h\_flip --save\_path "C:\\path\\to\\saveFolder" --include\_sub True --preview\_img***



The above configuration sets the target folder to 'C:\\path\\to\\target' and perform a vertical flip, followed by a horizontal flip on all of the images and labels in the target folder. The save path is 'C:\\path\\to\\saveFolder', and v\_flip should be ran on all subdirectories.



***python augmentify\_cli.py "C:\\path\\to\\target" h\_flip***



The above configuration sets the target folder to 'C:\\path\\to\\target' and performs a horizontal flip on all images and labels in the target folder



***python augmentify\_cli.py "C:\\path\\to\\target" add\_empty\_labels***



The above configuration sets the target folder to 'C:\\path\\to\\target' and creates an empty label file for each image in the folder

