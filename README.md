Augmentify is a CLI data augmentation tool used to assist in YOLO model training. Features include: flipping images vertically/horizontally, image filters, rotation, scaling/zooming, translations/shifting, padding, empty labels for negative data, and more in future development.



#### **How to install (Windows 11)**

First, ensure you have the necessary libraries: pip install albumentations opencv-python

1. Open a shell
2. cd into your preferred directory to install the program
3. Once in the preferred directory, run: 'git clone https://github.com/AlexTseng01/Augmentify.git'



You can now run Augmentify while inside the cloned directory.

You can also choose to run Augmentify globally in any directory by following the instructions below:



1. Press Win + R and type: "SystemPropertiesAdvanced" (Or press Win and type: "Environment Variables")
2. In System Properties, click on 'Environment Variables' on the bottom right of the window
3. Under 'User variables' click on 'Path', then click 'Edit'
4. Click on 'New' and paste the path of the cloned directory. It should look like: "C:\\path\\to\\Augmentify"
5. Press enter



You can now run Augmentify globally in any directory



#### **How to configure Augmentify**



After you have your terminal session with Augmentify installed, there are several ways you can configure it.

Augmentify requires two mandatory parameters to run; the target folder path that you want to augment, and one action to perform on the entire folder.



The target folder can be any folder as long as Augmentify has permission to access it.



The action can be any one of these listed:

* h\_flip
* v\_flip
* filter
* rotate
* scale
* shift
* pad
* add\_empty\_labels



There are three *optional* parameters you can include. It is *not* required and will run regardless of it being provided or not:

* save\_path accepts a path to a save folder
* include\_sub is a Boolean, accepting True or False
* preview\_img accepts a path to an image file



**Examples of configurations**:



***python augmentify\_cli.py "C:\\path\\to\\target" v\_flip --save\_path "C:\\path\\to\\saveFolder" --include\_sub True --preview\_img***



The above configuration sets the target folder to 'C:\\path\\to\\target' and perform a vertical flip on all of the images and labels in the target folder. The save path is 'C:\\path\\to\\saveFolder', and v\_flip should be ran on all subdirectories.



***python augmentify\_cli.py "C:\\path\\to\\target" h\_flip***



The above configuration sets the target folder to 'C:\\path\\to\\target' and performs a horizontal flip on all images and labels in the target folder



***python augmentify\_cli.py "C:\\path\\to\\target" add\_empty\_labels***



The above configuration sets the target folder to 'C:\\path\\to\\target' and creates an empty label file for each image in the folder

