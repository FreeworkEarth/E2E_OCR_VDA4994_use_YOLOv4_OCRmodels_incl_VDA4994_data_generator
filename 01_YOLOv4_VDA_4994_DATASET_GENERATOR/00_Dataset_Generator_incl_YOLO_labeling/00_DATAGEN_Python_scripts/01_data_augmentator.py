#!/usr/bin/env python
# above line to execute in linux
"""ONE FILE TO LABEL THEM ALL"""


""" Script needs to be in same folder as images!!!!"""

""" image augmentator: 
1)load base image ===> 
2)create augmented ===> 
3)put boxes in augmented includign labels 
4)==> save file with label name and augmented option at the end
5) save all files in dri in all lines in txt file"""

"""http://zetcode.com/python/pathlib/"""
""" create label dataset for training"""

from matplotlib import pyplot as plt
import numpy as np
import cv2
import random
import string
import os
from PIL import ImageFont, ImageDraw, Image
from pathlib import Path, PurePath
import pathlib

""" Create random image loader: """

## Set 4K or FullHD for image augmentator
FullHD = 0;   # one for FullHD dataset, else 4K normal res

# Sort Paths
print(f"Current directory: {Path.cwd()}")
print(f"Home directory: {Path.home()}")
path_to_training_images = Path("00_Dataset_Generator_incl_YOLO_labeling\4k_images\00_Raw_images")
print(path_to_training_images.__hash__())
print(path_to_training_images)                  # windows path
print(path_to_training_images.as_posix())       # linux path



""" If images in same directory as script"""
path = Path.cwd()
print(os.listdir(path))
"""List all paths of folder in list"""
files = [e for e in path.iterdir() if e.is_file()]
print(files)

 #Todo search #images in list and use image augmentator for each of the basic objects



"""TODO random choice of file in rootpath of python script"""
random_filename = random.choice([x for x in os.listdir(path)if os.path.isfile(x)])
print(random_filename)
random_image = cv2.imread(random_filename)

"""Todo Rescale image if demanded to FullHD"""
if FullHD is 1:
    resized_image = cv2.resize(random_image, (1920, 1080))
    print(resized_image)
    #plt.imshow(resized_image)
    cv2.imshow(str(random_filename), resized_image)
    cv2.waitKey(10000000)
    cv2.destroyAllWindows()
else:
    print(random_image)
    cv2.imshow(str(random_filename), random_image)
    cv2.waitKey(10000000)
    cv2.destroyAllWindows()

""" print 4k AND FullHD"""
# cv2.imshow(str(random_filename), random_image)
# cv2.waitKey(10000000)
# cv2.destroyAllWindows()
# resized_image = cv2.resize(random_image, (1920, 1080))
# cv2.imshow(str(random_filename), resized_image)
# cv2.waitKey(10000000)
# cv2.destroyAllWindows()


