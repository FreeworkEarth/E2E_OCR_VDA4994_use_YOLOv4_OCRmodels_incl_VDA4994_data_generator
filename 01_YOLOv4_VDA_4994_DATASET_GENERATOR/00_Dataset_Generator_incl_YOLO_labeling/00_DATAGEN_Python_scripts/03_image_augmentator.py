from matplotlib import pyplot as plt
import numpy as np
import cv2
import random
import string
import os
from PIL import ImageFont, ImageDraw, Image
from pathlib import Path, PurePath
import pathlib
import Augmentor


path_to_training_images = Path(r"C:\Users\chari\Google Drive\00_Masterthesis\Masterarbeit\10_MA_COde\Dataset\00_Datset_Generator_images\4k_Wallpaper images\02_Test_Image_augmentator")
print(path_to_training_images)

# r before string or / instead of \
p = Augmentor.Pipeline("C:/Users/chari/Google Drive/00_Masterthesis/Masterarbeit/10_MA_COde/Dataset/00_Datset_Generator_images/4k_Wallpaper images/02_Test_Image_augmentator")
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
print(p)




