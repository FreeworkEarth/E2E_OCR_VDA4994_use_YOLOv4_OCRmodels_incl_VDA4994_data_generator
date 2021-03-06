#!/usr/bin/env python
# above line to execute in linux
"""ONE FILE TO LABEL THEM ALL"""
import numpy as np
import cv2
import random
import string
import pathlib
import shutil
from pathlib import Path
from pathlib import PurePath
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path, PurePath
from glob import glob
import glob
import os
# fonts
from PIL import ImageFont, ImageDraw, Image
# random text/numbers
from random import randrange, uniform, randint
#from Helper_Functions.random_text_gen import random_text_gen
#from Helper_Functions import random_numbers_strings
import random_text_gen
from random_text_gen import *
import qrcode



"""WORKFLOW: Training and Testdata generator (not Augmentator so far): 
0. GIVE Absolute path of raw image folder == List all filenames and LOAD just Images in order of list 
1. label size definition
2. 4K/ FullHD 
3. Load image
3. loop for original image or augmentation
4. random VDA label integration (yellow and white) including random strings/ labels AND LABELS for YOLOv3
5. save all images with specific label names"""

""" BUILIDNG BLOCK DRAW VDA LABEL including YOLO Bounding Boxes"""
""" 1. define yellow label START pixels"""
""" 2. draw yellow label / box (without YOLO)"""
""" 3. define text  for yellow label """
""" 4. define Labels text positions"""
""" 5. draw text for labels """

""" YOLO text (text = classname)"""
""" 6. define YOLO BOXES AND TEXT position"""
""" 7. draw YOLO bounding boxes"""
""" 8. Write classname into boxes"""
""" 9 . normalize for labelling"""


# TODO while loop which creates random labels in an image
# todo create label boxes around createt labels for YOLO including random colours (rgb 0-255) = 8 bit
# TODO all open cv image operations/ data augementator ops and randomly choose a few inside a while loop for a label position ( for each label ==> picture in different variations)

""" SETUP VARIABLES to run SCRIPT"""
# yolo version
yolo_version = 4
# image path setup ( here: in 01_4k_images_raw/00_Raw_images_4k)
path_to_training_images = Path(r"C:\Users\chari\Google Drive\00_Masterthesis\Masterarbeit\000_CODE\Git\01_YOLOv4_VDA_4994_DATASET_GENERATOR\00_Dataset_Generator_incl_YOLO_labeling\01_4k_images_raw\xxx_Test_for_Image_generator")  ## ipath to raw images from repo
image_resolution = "FullHD"     #"FullHD"  or 4K      ____4096 × 2160 (full frame, 256∶135 or ≈1.90∶1 aspect ratio)

number_datasets = 5
complete_dataset = [[]]
complete_datset_np = np.array([],[])

labels_random_size_ratio_min = 0.2          # min 0.1
labels_random_size_ratio_max = 10         # max 10
alphanumeric_random_size_ratio_min = 0.1
alphanumeric_random_size_ratio_max = 2

labels_random_size_ratio_min_FullHD = labels_random_size_ratio_min / 2
labels_random_size_ratio_max_FullHD = labels_random_size_ratio_max / 2
alphanumeric_random_size_ratio_min_FullHD = alphanumeric_random_size_ratio_min/2
alphanumeric_random_size_ratio_max_FullHD = alphanumeric_random_size_ratio_max/2

font = cv2.FONT_HERSHEY_SIMPLEX  #OpenCV font
number_of_images_per_basic_image = 5  # how many images are printed for each backround image
white_depend_on_yellow_label = False   # if white and yellow label are always in the same difference to each other

wht_col_R = 255
wht_col_G = 255
wht_col_B = 255
black_col_R = 0
black_col_G = 0
black_col_B = 0

""" YOLOv3 Label , VDA label and picture size definition"""
## yellow VDA label:
yl_label_width_no_scaled_4k = int(225)
yl_label_height_no_scaled  = int(50)
## white VDA label A5 size or ratio = 1.4189
wht_label_width_no_scaled  = int(210)
wht_label_height_no_scaled  = int(148)
#""" Define distance of white and yellow pixels ==> use in defining random start pixel area """
wht_dstnc_l_t_x_no_scaled  = int(25)
wht_dstnc_l_t_y_no_scaled  = int(100)
#"""YOLOv3 label ==> like label from https://pjreddie.com/darknet/yolo/ """
# Todo draw box around yellow and white label including name == random string inside labels
linestrength_YOLOv3_label_no_scaled  = int(2)
heigth_naming_YOLOv3_label_no_scaled  = int(25)

""" CLASSES setup for custom VDA 4994 YOLOv3/YOLOv4 labelled (configured) dataset"""

class_name_yellow_VDA = "yellow_VDA_label"
class_number_yellow_VDA = 0
class_name_white_VDA_9449 = "white_VDA_label"
class_number_white_VDA = 1

class_names = []
class_names.append(class_name_yellow_VDA)
class_names.append(class_name_white_VDA_9449)
number_of_classes = len(class_names)
#number_of_classes_selfcount =

""" get all image elements in current directory """
# Sort Paths
current_working_dir = f"Current directory: {Path.cwd()}"
home_dir = f"Home directory: {Path.home()}"
# define absolute path to raw image dataset


# prints of paths
print(path_to_training_images.__hash__())
print(path_to_training_images)                  # windows path
print(path_to_training_images.as_posix())       # linux path


# list of images in raw image path as jpg
# just .jpg files
img_list = []
img_list_path_absolute = []
for r, d, f in os.walk(path_to_training_images):
    for file in f:
        if file.endswith(".jpg"):
            #print(os.path.join(r, file))
            img_list.append(file)
            # get absolute paths of images
            img_list_path_absolute.append(str(path_to_training_images) + "/" + file)

print(img_list)
print(len(img_list))
print(img_list[0])
print(img_list_path_absolute)
print(len(img_list_path_absolute))
print(str(img_list_path_absolute[0]))

## all files and elements in path
# img_list_2 = []
# arr = os.listdir(path_to_training_images)
# for file in arr:
#     img_list_2.append(file)
# print(img_list_2)

""" HELPER FUNCTIONS"""
#""" text generator"""
# def random_text_gen(length=32, randomascii=True, uppercase=True, lowercase=True, numbers=True):
#     character_set = ''  # lowercase, uppercase, digits etc. possible
#     if randomascii:
#         character_set += string.ascii_letters
#     elif uppercase:
#         character_set += string.ascii_uppercase
#     elif lowercase:
#         character_set += string.ascii_lowercase
#     elif numbers:
#         character_set += string.digits
#     return ''.join(random.choice(character_set) for i in range(length))

# """ ALL FONTS """
# Todo enable new fonts
# https://www.codesofinterest.com/2017/07/more-fonts-on-opencv.html

# ft = cv2.freetype.createFreeType2()
# ft.loadFontData(fontFileName='Ubuntu-R.ttf',
#                 id=0)
# ft.putText(img=img,
#            text='Quick Fox',
#            org=(15, 70),
#            fontHeight=60,
#            color=(255, 255, 255),
#            thickness=-1,
#            line_type=cv2.LINE_AA,
#            bottomLeftOrigin=True)

""" LOOP to Create VDA labels inside random base image"""
# loop over all images in directory = load image [i]
for i in range(len(img_list_path_absolute)):

    #" n different labels per image"
    count = 0
    while count < number_of_images_per_basic_image:

        # Image Size's 4K 16:9 UHD TV standard and Full HD
        fourK_width = 3840
        fourK_height = 2160
        FullHD_width = int(fourK_width / 2)
        FullHD_height = int(fourK_height / 2)

        """load image"""
        img = cv2.imread(img_list_path_absolute[i])

        """resize image if should be FullHD"""
        if image_resolution is "4K":
            pass
        else:
            dim_fullHD = (FullHD_width, FullHD_height)
            img = cv2.resize(img, dim_fullHD, interpolation=cv2.INTER_AREA)
            labels_random_size_ratio_min = labels_random_size_ratio_min_FullHD
            labels_random_size_ratio_max = labels_random_size_ratio_max_FullHD
            alphanumeric_random_size_ratio_min = alphanumeric_random_size_ratio_min_FullHD
            alphanumeric_random_size_ratio_max = alphanumeric_random_size_ratio_max_FullHD

        x = img_list_path_absolute[i]
        print(x)

        """ Show image size """
        ##img = cv2.imread('../WC9 - 390Y.jpg')
        width_img, height_img, channels_img = img.shape[1], img.shape[0], img.shape[2]
        print(img.shape, "\n width image:{} pixel \n heigth image:{} pixel \n channels image: {} channels".format(width_img, height_img, channels_img))

        """scaling (randomly) of VDA labels"""
        ## label size factors
        factor_size_labels = random.uniform(labels_random_size_ratio_min, labels_random_size_ratio_max)
        factor_size_alphanumerics = random.uniform(alphanumeric_random_size_ratio_min, alphanumeric_random_size_ratio_max)
        print("factor size labels = {}".format(factor_size_labels))
        print("factor size alphas = {}".format(factor_size_alphanumerics))


        """labels"""
        ## yellow VDA label:
        yl_label_width = int(yl_label_width_no_scaled_4k * factor_size_labels)
        yl_label_height = int(yl_label_height_no_scaled * factor_size_labels)
        ## white VDA label
        wht_label_width = int(wht_label_width_no_scaled * factor_size_labels)
        wht_label_height = int(wht_label_height_no_scaled * factor_size_labels)
        # """ Define distance of white and yellow pixels ==> use in defining random start pixel area """
        wht_dstnc_l_t_x = int(wht_dstnc_l_t_x_no_scaled * factor_size_labels)
        wht_dstnc_l_t_y = int(wht_dstnc_l_t_y_no_scaled * factor_size_labels)
        # """YOLO label ==> like label from https://pjreddie.com/darknet/yolo/ """
        # Todo draw box around yellow and white label including name == random string inside labels
        linestrength_YOLO_label = int(linestrength_YOLOv3_label_no_scaled * factor_size_labels)
        heigth_naming_YOLO_label = int(heigth_naming_YOLOv3_label_no_scaled * factor_size_labels)
        #Todo: ensure no overlappings? Ensure label including full bounding box inside generated picture


        """Start drawing labels at random places but always inside picture (including factorization)
        FOR EACH LABEL SEPARATED"""
        """Calculated by hand (Case diffenece between 4k and FullHD"""
        ##Todo: random start of yellow label ===> decide whether 4K/FullHD ===> 4k between 0 and 3840-yl_label
        #"""Differences in width and height 4K"""
        # fourK_width_diff = fourK_width - yl_label_width + 2 * linestrength_YOLO_label
        # fourK_height_diff = fourK_height - (yl_label_height + heigth_naming_YOLO_label * 2 + linestrength_YOLO_label) #(yl_label_height + wht_dstnc_l_t_y + wht_label_height + heigth_naming_YOLO_label + linestrength_YOLO_label)
        # """Differences in width and height FullHD"""
        # FullHD_width_diff = FullHD_width - yl_label_width + 2 * linestrength_YOLO_label
        # FullHD_height_diff = FullHD_height - (yl_label_height + wht_dstnc_l_t_y + wht_label_height + heigth_naming_YOLO_label + linestrength_YOLO_label)
        # print(fourK_width_diff,fourK_height_diff,FullHD_width_diff, FullHD_height_diff)
        """starting pixels"""
        # start_pxl_lft_tp_x = int(float(random.uniform(0, FullHD_width_diff)))
        # start_pxl_lft_tp_y = int(float(random.uniform(0, FullHD_height_diff)))
        # print("start pixel x is: {} and start pixel in y is: {}".format(start_pxl_lft_tp_x , start_pxl_lft_tp_y))


        """Auto Calculate Difference from loaded image (WITH OpenCV ==>> height, witdth channels) from shape (=size)"""
        auto_calc_start_pxl_lft_tp_x_range_yellow = img.shape[1] - (yl_label_width + 2 * linestrength_YOLO_label)
        auto_calc_start_pxl_lft_tp_y_range_yellow = img.shape[0] -((yl_label_height + heigth_naming_YOLO_label * 2 + linestrength_YOLO_label))
        auto_calc_start_pxl_lft_tp_x_range_white = img.shape[1] - (yl_label_width + 2 * linestrength_YOLO_label)
        auto_calc_start_pxl_lft_tp_y_range_white = img.shape[0] - ((wht_label_height + heigth_naming_YOLO_label * 2 + linestrength_YOLO_label))
        print(auto_calc_start_pxl_lft_tp_x_range_yellow, auto_calc_start_pxl_lft_tp_y_range_yellow, auto_calc_start_pxl_lft_tp_x_range_white,auto_calc_start_pxl_lft_tp_y_range_white)

        ## dependet on white label height
        #auto_calc_start_pxl_lft_tp_y_range_white = img.shape[0] - (yl_label_height + wht_dstnc_l_t_y + wht_label_height + heigth_naming_YOLO_label * 2 + linestrength_YOLO_label)

        #"""TEST === > after in if conditional start fullHD"""
        auto_calc_start_xl_lft_tp_x_yellow = int(float(random.uniform(0, auto_calc_start_pxl_lft_tp_x_range_yellow)))
        auto_calc_start_xl_lft_tp_y_yellow = int(float(random.uniform(0, auto_calc_start_pxl_lft_tp_y_range_yellow)))

        auto_calc_start_xl_lft_tp_x_white = int(float(random.uniform(0, auto_calc_start_pxl_lft_tp_x_range_white)))
        auto_calc_start_xl_lft_tp_y_white = int(float(random.uniform(0, auto_calc_start_pxl_lft_tp_y_range_white)))
        print(auto_calc_start_xl_lft_tp_x_yellow, auto_calc_start_xl_lft_tp_y_yellow, auto_calc_start_xl_lft_tp_x_white, auto_calc_start_xl_lft_tp_y_white)

        # """    CLASSES:    Generalized for alphanumeric"""
        # # ## write classes:
        # #
        # #
        # #
        # # ### id_alphanumeric
        # # amount_id_alphanumeric = 35  # 0-9 and a-z = 36 alphanumeric numbers starting with 0
        # # ### random id
        # # random_id_alphanumeric = int(float(random.uniform(0,amount_id_alphanumeric)))
        # # print(random_id_alphanumeric)
        # #
        # # size_alphanumeric_generic = 1
        # # factor_size_alphanumeric = random.uniform(labels_random_size_ratio_min, labels_random_size_ratio_max)
        # # printsize_alphanumeric = size_alphanumeric_generic * factor_size_alphanumeric
        # #
        # #
        # # ### print alphanumerics
        # # counter_number_alphanumeric = 5
        # # while i < counter_number_alphanumeric:
        # #
        # #     #alphanumeric_print = alphanumeric_dict{radnom_id_alphanumeric}
        # #
        # #     ### PRINT IN TXT
        # #         #f.write in textfile of classes for bounding box
        # #     #### if random_id_alphanumeric == random_id_alphanumeric)
        # #
        # #     pass
        # # ### size alphanumeric
        #
        # dict_classes_alphanumeric = {}
        # string_alphanumeric_numbers = "0123456789"
        # string_alphanumeric_lowercase = "abcdefghijklmnopqrstuvwxyz"
        # string_alphanumeric_uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        #
        # string_alphanumeric = string_alphanumeric_numbers + string_alphanumeric_uppercase + string_alphanumeric_lowercase
        # number_classes_alphanumeric = len(string_alphanumeric)
        # print(number_classes_alphanumeric)
        # print(len(string_alphanumeric))
        # print(string_alphanumeric)
        #
        # class_names_list = []
        # class_values_list_string = []
        # class_values_list = []
        #
        # # """ all through dictionary = dynamic"""
        # class_names_list_bydict = []
        # class_values_list_bydict = []
        # class_names = dict_classes_alphanumeric.keys()
        # class_values = dict_classes_alphanumeric.values()
        #
        # cnt_string_alphnmbr = 0
        # for cnt_string_alphnmbr in range(len(string_alphanumeric)):
        #     dict_classes_alphanumeric[cnt_string_alphnmbr] = string_alphanumeric[cnt_string_alphnmbr]
        #     number_classes = len(dict_classes_alphanumeric)
        #     class_names_list.append("class_name_{}".format(cnt_string_alphnmbr))
        #     class_values_list_string.append(string_alphanumeric[cnt_string_alphnmbr])
        #
        # for key, value in dict_classes_alphanumeric.items():
        #     class_names_list_bydict.append(key)
        #     # class_names_list_solo_int.append"{}"
        #     class_values_list_bydict.append(dict_classes_alphanumeric[key])
        #
        # dict_classes_alphanumeric_tuple = dict_classes_alphanumeric.items()
        # print(dict_classes_alphanumeric_tuple)
        # print(class_names_list)
        # print(class_values_list)
        # print(class_names)
        # print(class_values)
        #
        # ######
        # print(class_names_list_bydict)
        # print(class_values_list_bydict)
        # print(dict_classes_alphanumeric)
        #
        # """ print random alphanumeric sign into loaded picture"""
        # number_printed_alphanumerical = 50
        #
        # nmbr_alphanum = 0
        # for nmbr_alphanum in range(number_printed_alphanumerical):
        #     random_alphanumerical_key = random.randint(0, len(class_names_list_bydict) - 1)
        #     # print(random_alphanumerical_key)
        #
        #     value_random_alphanumerical = dict_classes_alphanumeric[random_alphanumerical_key]
        #     print(value_random_alphanumerical)
        #
        #     ### starting left top alphanumeric
        #     auto_calc_start_alphanumeric_left_top_x = int(float(random.uniform(0, img.shape[1])))
        #     auto_calc_start_alphanumeric_left_top_y = int(float(random.uniform(0, img.shape[0])))
        #
        #     #### print random letter in image
        #     """ Draw radnom alphanumerics in image """
        #     ### yellow label
        #     ## open CV
        #     # cv2.putText(img, label_yellow_text_combined, (txt_pos_blyl_x, txt_pos_blyl_y), font, 0.5 * factor_size_labels, (0, 0, 0),
        #     #             2, cv2.LINE_AA)
        #     # Convert the image to RGB (OpenCV uses BGR)
        #     cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     # Pass the image to PIL
        #     pil_im = Image.fromarray(cv2_im_rgb)
        #     draw = ImageDraw.Draw(pil_im)
        #     font_size_blyl = 200
        #     # use a truetype font
        #     factor_size_alphanumerics = random.uniform(alphanumeric_random_size_ratio_min,
        #                                                alphanumeric_random_size_ratio_max)
        #
        #     font_pil = ImageFont.truetype("arial.ttf", int(font_size_blyl * factor_size_alphanumerics))
        #     # Draw the text
        #     draw.text((auto_calc_start_alphanumeric_left_top_x,
        #                auto_calc_start_alphanumeric_left_top_y - factor_size_alphanumerics * font_size_blyl),
        #               value_random_alphanumerical, fill="black", font=font_pil)
        #     # Get back the image to OpenCV
        #     img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        """
        CLASS 0
        Yellow label  dependend on random image pixels in range of image pixels
              class number 0 for YOLO labelling
        """

        def draw_yellow_VDA_label():
            pass


        """ 1. define yellow label pixels"""
        yl_pxl_l_t_x = auto_calc_start_xl_lft_tp_x_yellow      # start_pxl_lft_tp_x
        yl_pxl_l_t_y = auto_calc_start_xl_lft_tp_y_yellow      # start_pxl_lft_tp_y
        yl_pxl_r_b_x = yl_pxl_l_t_x + yl_label_width    # yellow pixel right bottom x
        yl_pxl_r_b_y = yl_pxl_l_t_y + yl_label_height   # yellow pixel right bottom y
        yl_col_R = 255
        yl_col_G = 255
        yl_col_B = int(0)
        ## black box inside yellow label depends on yellow label
        bb_pxl_l_t_x = yl_pxl_r_b_x - yl_label_height
        bb_pxl_l_t_y = yl_pxl_r_b_y - yl_label_height
        bb_pxl_r_b_x = yl_pxl_r_b_x
        bb_pxl_r_b_y = yl_pxl_r_b_y


        """ 2. draw yellow label / box (without YOLO)"""
        # Yellow VDA label box
        img = cv2.rectangle(img, (int(float(yl_pxl_l_t_x)), int(float((yl_pxl_l_t_y)))),
                            (int(float(yl_pxl_r_b_x)), int(float(yl_pxl_r_b_y))),
                            (yl_col_B, yl_col_G, yl_col_R), -1)
        # Black inside Yellow label box
        img = cv2.rectangle(img, (bb_pxl_l_t_x, bb_pxl_l_t_y), (bb_pxl_r_b_x, bb_pxl_r_b_y),
                            (0, 0, 0), -1)



        """ 3. define text  for yellow label """
        # """ random label text generator"""
        #  yellow label
        label_1st_pos = random_text_gen(2, randomascii=False, uppercase=True)
        label_2nd_pos = random_text_gen(1, randomascii=False, uppercase=False, lowercase=False)
        label_3rd_pos = '-'
        label_4th_pos = random_text_gen(3, randomascii=False, uppercase=False, lowercase=False)
        label_5th_pos = random_text_gen(1, randomascii=False, uppercase=True)
        label_yellow_text_combined = '%s%s %s %s%s' % (label_1st_pos, label_2nd_pos, label_3rd_pos, label_4th_pos, label_5th_pos)
        print(label_yellow_text_combined)


        """ 4. define yellow Label Labels text positions"""
        # text position inside yellow label
        txt_distance_x = int(40 * factor_size_labels)
        txt_distance_y = int(5 * factor_size_labels)
        txt_pos_blyl_x = yl_pxl_l_t_x + txt_distance_x  # pixel left bottom x
        txt_pos_blyl_y = yl_pxl_l_t_y + yl_label_height - txt_distance_y  # pixel left bottom y
        # on black box
        txt_distance_yellow_x = int(6 * factor_size_labels)
        txt_distance_yellow_y = int(12 * factor_size_labels)
        txt_pos_ylbl_x = bb_pxl_l_t_x + txt_distance_yellow_x
        txt_pos_ylbl_y = bb_pxl_l_t_y + yl_label_height - txt_distance_yellow_y

        """ 5. draw text inside yellow  labels """
        ### yellow label
        ## open CV
        # cv2.putText(img, label_yellow_text_combined, (txt_pos_blyl_x, txt_pos_blyl_y), font, 0.5 * factor_size_labels, (0, 0, 0),
        #             2, cv2.LINE_AA)

        # Convert the image to RGB (OpenCV uses BGR)
        cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Pass the image to PIL
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        font_size_blyl = 20
        # use a truetype font
        font_pil = ImageFont.truetype("arial.ttf", int(font_size_blyl * factor_size_labels))
        # Draw the text
        draw.text((txt_pos_blyl_x, txt_pos_blyl_y - factor_size_labels*font_size_blyl), label_yellow_text_combined, fill="black", font=font_pil)
        # Get back the image to OpenCV
        img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


        ### black number
        ## OpenCv
        # cv2.putText(img, random_text_gen(2, randomascii=False, uppercase=False, lowercase=False),
        #             (txt_pos_ylbl_x, txt_pos_ylbl_y), font, 1 * factor_size_labels, (yl_col_B, yl_col_G, yl_col_R), 2,
        #             cv2.LINE_AA)

        ## pillow
        # Convert the image to RGB (OpenCV uses BGR)
        cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Pass the image to PIL
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        size_font_ylbl = 35
        # use a truetype font
        font_pil = ImageFont.truetype("arial.ttf", int(size_font_ylbl * factor_size_labels))
        # Draw the text
        draw.text((txt_pos_ylbl_x, txt_pos_ylbl_y-factor_size_labels*size_font_ylbl), random_text_gen(2, randomascii=False, uppercase=False, lowercase=False), fill="yellow", font=font_pil)
        # Get back the image to OpenCV
        img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)



        """YOLO text in bounding box (text ==== classname)"""
        """ 6. Define YOLO BOXES AND TEXT position in bounding box """
        ## YOLOv3 Label for yellow box depends on size of yellow box
        YOLOv3_pxl_l_t_x = auto_calc_start_xl_lft_tp_x_yellow - linestrength_YOLO_label
        YOLOv3_pxl_l_t_y = auto_calc_start_xl_lft_tp_y_yellow - linestrength_YOLO_label
        YOLOv3_pxl_r_b_x = YOLOv3_pxl_l_t_x + yl_label_width + linestrength_YOLO_label
        YOLOv3_pxl_r_b_y = YOLOv3_pxl_l_t_y + yl_label_height + linestrength_YOLO_label
        # naming box
        YOLOv3_pxl_l_t_x_name = YOLOv3_pxl_l_t_x
        YOLOv3_pxl_l_t_y_name = YOLOv3_pxl_l_t_y - heigth_naming_YOLO_label
        YOLOv3_pxl_r_b_x_name = YOLOv3_pxl_r_b_x - yl_label_height
        YOLOv3_pxl_r_b_y_name = YOLOv3_pxl_l_t_y
        # colour YOLO yellow label
        R_value_YOLOv3_label = random.randint(0, 255)
        G_value_YOLOv3_label = random.randint(0, 255)
        B_value_YOLOv3_label = random.randint(0, 255)
        # on black box
        txt_distance_yellow_x = int(6 * factor_size_labels)
        txt_distance_yellow_y = int(12 * factor_size_labels)
        txt_pos_ylbl_x = bb_pxl_l_t_x + txt_distance_yellow_x
        txt_pos_ylbl_y = bb_pxl_l_t_y + yl_label_height - txt_distance_yellow_y
        #  YOLOv3 label on yellow VDA label
        txt_yolo_yl_dstnc_x = int(3 * factor_size_labels)
        txt_yolo_yl_dstnc_y = int(3 * factor_size_labels)
        txt_pos_yl_yolov3_label_x = YOLOv3_pxl_l_t_x + txt_yolo_yl_dstnc_x
        txt_pos_yl_yolov3_label_y = YOLOv3_pxl_l_t_y - txt_yolo_yl_dstnc_y

        """ 7. Draw YOLO bounding boxes"""
        # YOLOv3 bounding box Yellow Label
        img = cv2.rectangle(img, (YOLOv3_pxl_l_t_x, YOLOv3_pxl_l_t_y), (YOLOv3_pxl_r_b_x, YOLOv3_pxl_r_b_y),
                            (B_value_YOLOv3_label, G_value_YOLOv3_label, R_value_YOLOv3_label),
                            thickness=linestrength_YOLO_label)
        # naming box
        img = cv2.rectangle(img, (YOLOv3_pxl_l_t_x_name, YOLOv3_pxl_l_t_y_name),
                            (YOLOv3_pxl_r_b_x_name, YOLOv3_pxl_r_b_y_name),
                            (B_value_YOLOv3_label, G_value_YOLOv3_label, R_value_YOLOv3_label), -1)

        """8. Write classname into boxes"""
        # # yellow YOLO label
        # cv2.putText(img, class_name_0, (txt_pos_yl_yolov3_label_x, txt_pos_yl_yolov3_label_y), font,
        #             0.5 * factor_size_labels, (0, 0, 0), 2, cv2.LINE_AA)

        ## pillow
        # Convert the image to RGB (OpenCV uses BGR)
        cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Pass the image to PIL
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        size_font_yolo_label_yellow = 15
        # use a truetype font
        font_pil = ImageFont.truetype("arial.ttf", int(size_font_yolo_label_yellow * factor_size_labels))
        # Draw the text
        draw.text((txt_pos_yl_yolov3_label_x, txt_pos_yl_yolov3_label_y - factor_size_labels * size_font_yolo_label_yellow),
                  class_name_yellow_VDA, fill="black", font=font_pil)


        # Get back the image to OpenCV
        img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)





        """ BUILIDNG BLOCK DRAW  white VDA LABEL including YOLO Bounding Boxes"""
        """ 1. define yellow label START pixels"""
        """ 2. draw yellow label / box (without YOLO)"""
        """ 3. define text  for  label """
        """ 4. define Labels text positions"""
        """ 5. draw text for labels """
        """ YOLO text (text = classname)"""
        """ 6. define YOLO text position"""
        """ 7. draw YOLO bounding boxes"""
        """ 8. Write classname into boxes"""



        """CLASS 1"""
        """white VDA label DIN 4994 https://www.my-vda-label.de/VDA-Label-drucken-4994 dependent on dependent/ independent start pixel
        class number 1 for YOLO labelling  
        https://www.my-vda-label.de/VDA-Label-drucken-4994
        https://label.tec-it.com/de/Group/VDA4994/VDA_4994_DE_A5
        file:///C:/Users/chari/Downloads/VDA_4994_(Deutsch)_A5.PDF """

        def draw_white_VDA_label():
            pass

        """ 1. define white START  label pixels"""
        ### dependent / independet  of yellow label"""
        if white_depend_on_yellow_label is True:
            wht_pxl_l_t_x = auto_calc_start_xl_lft_tp_x_white + wht_dstnc_l_t_x                      # pixel left top white box
            wht_pxl_l_t_y = auto_calc_start_xl_lft_tp_y_white + wht_dstnc_l_t_y
            wht_pxl_r_b_x = wht_pxl_l_t_x + wht_label_width
            wht_pxl_r_b_y = wht_pxl_l_t_y + wht_label_height
            wht_col_R = 255
            wht_col_G = 255
            wht_col_B = 255
        else:
            wht_pxl_l_t_x = auto_calc_start_xl_lft_tp_x_white       # + wht_dstnc_l_t_x  # pixel left top white box
            wht_pxl_l_t_y = auto_calc_start_xl_lft_tp_y_white       # + wht_dstnc_l_t_y
            wht_pxl_r_b_x = wht_pxl_l_t_x + wht_label_width
            wht_pxl_r_b_y = wht_pxl_l_t_y + wht_label_height
            wht_col_R = 255
            wht_col_G = 255
            wht_col_B = 255
            black_col_R = 0
            black_col_G = 0
            black_col_B = 0


        """ 2. draw white label / box"""
        ## draw White VDA label including lines etc.
        img = cv2.rectangle(img, (wht_pxl_l_t_x, wht_pxl_l_t_y), (wht_pxl_r_b_x, wht_pxl_r_b_y),
                            (wht_col_B, wht_col_G, wht_col_R), -1)

        ## lines inside white label
        thickness_line_wht_label = 5
        ##horizontal lines y - values
        dstnce_first_hor = int(40 * factor_size_labels)
        dstnce_scnd_hor = int(dstnce_first_hor + 17 * factor_size_labels)
        dstcne_thrd_hor = int(dstnce_scnd_hor + 17 * factor_size_labels)
        dstcne_fourth_hor = int(dstcne_thrd_hor + 37 * factor_size_labels)
        dstcne_fifth_hor = int(dstcne_fourth_hor + 37 * factor_size_labels)

        ## vertical lines y- values
        dstnce_first_ver = int(60 * factor_size_labels)
        dstnce_scnd_ver =  int(dstnce_first_ver + 90 * factor_size_labels)
        dstcne_thrd_ver = int(dstnce_scnd_ver + 7 * factor_size_labels)
        dstcne_fourth_ver = int(dstcne_thrd_ver - 20 * factor_size_labels)


        ## horizontal lines
        img = cv2.line(img, (wht_pxl_l_t_x, wht_pxl_l_t_y + dstnce_first_hor), (wht_pxl_r_b_x, wht_pxl_l_t_y + dstnce_first_hor),
                            (black_col_B, black_col_G, black_col_R), thickness=thickness_line_wht_label )
        img = cv2.line(img, (wht_pxl_l_t_x, wht_pxl_l_t_y + dstnce_scnd_hor),
                       (wht_pxl_r_b_x, wht_pxl_l_t_y + dstnce_scnd_hor),
                       (black_col_B, black_col_G, black_col_R), thickness=thickness_line_wht_label)
        img = cv2.line(img, (wht_pxl_l_t_x, wht_pxl_l_t_y + dstcne_thrd_hor),
                       (wht_pxl_r_b_x, wht_pxl_l_t_y + dstcne_thrd_hor),
                       (black_col_B, black_col_G, black_col_R), thickness=thickness_line_wht_label)
        img = cv2.line(img, (wht_pxl_l_t_x, wht_pxl_l_t_y + dstcne_fourth_hor),
                       (wht_pxl_r_b_x, wht_pxl_l_t_y + dstcne_fourth_hor),
                       (black_col_B, black_col_G, black_col_R), thickness=thickness_line_wht_label)
        img = cv2.line(img, (wht_pxl_l_t_x, wht_pxl_l_t_y + dstcne_fifth_hor),
                       (wht_pxl_r_b_x, wht_pxl_l_t_y + dstcne_fifth_hor),
                       (black_col_B, black_col_G, black_col_R), thickness=thickness_line_wht_label)

        ## vertcal lines
        img = cv2.line(img, (wht_pxl_l_t_x + dstnce_first_ver, wht_pxl_l_t_y + dstnce_first_hor),
                       (wht_pxl_l_t_x + dstnce_first_ver, wht_pxl_l_t_y + dstnce_scnd_hor),
                       (black_col_B, black_col_G, black_col_R), thickness=thickness_line_wht_label)
        img = cv2.line(img, (wht_pxl_l_t_x + dstnce_scnd_ver, wht_pxl_l_t_y+ dstnce_first_hor),
                       (wht_pxl_l_t_x + dstnce_scnd_ver, wht_pxl_l_t_y  + dstnce_scnd_hor),
                       (black_col_B, black_col_G, black_col_R), thickness=thickness_line_wht_label)
        img = cv2.line(img, (wht_pxl_l_t_x + dstcne_thrd_ver, wht_pxl_l_t_y + dstcne_thrd_hor),
                       (wht_pxl_l_t_x + dstcne_thrd_ver, wht_pxl_l_t_y + dstcne_fourth_hor),
                       (black_col_B, black_col_G, black_col_R), thickness=thickness_line_wht_label)
        img = cv2.line(img, (wht_pxl_l_t_x + dstcne_fourth_ver, wht_pxl_l_t_y + dstcne_fourth_hor),
                       (wht_pxl_l_t_x + dstcne_fourth_ver, wht_pxl_l_t_y + dstcne_fifth_hor),
                       (black_col_B, black_col_G, black_col_R), thickness=thickness_line_wht_label)

        """ 3. define text for white label """
        # white label
        label_1st_pos = random_text_gen(3, randomascii=False, uppercase=True)
        label_2nd_pos = '-'
        label_3rd_pos =  random_text_gen(3, randomascii=False, uppercase=False, lowercase=False)
        label_4th_pos = '-'
        label_5th_pos = random_text_gen(3, randomascii=False, uppercase=False, lowercase=False)
        label_6th_pos = '-'
        label_7th_pos = random_text_gen(3, randomascii=False, uppercase=False, lowercase=False)
        label_white_number = '%s%s%s%s%s%s%s' % (label_1st_pos, label_2nd_pos, label_3rd_pos, label_4th_pos, label_5th_pos, label_6th_pos, label_7th_pos)
        #label_white_number = random_text_gen(10, randomascii=False, uppercase=False, lowercase=False)
        print(label_white_number)
        label_white_str = r'%s' % (label_white_number)

        """ 4. define Labels text positions"""
        # on white label
        txt_wh_dstnc_x = int(38 * factor_size_labels)
        txt_wh_dstnc_y = int(57 * factor_size_labels)
        txt_pos_wh_x = wht_pxl_l_t_x + txt_wh_dstnc_x
        txt_pos_wh_y = wht_pxl_l_t_y + txt_wh_dstnc_y

        """ 5. draw text for labels """
        # cv2.putText(img, label_white_number, (txt_pos_wh_x, txt_pos_wh_y), font, 0.8 * factor_size_labels, (0, 0, 0), 2,
        #             cv2.LINE_AA)

        """ other fonts"""
        ## https://www.codesofinterest.com/2017/07/more-fonts-on-opencv.html
        ## https: // pillow.readthedocs.io / en / stable / reference / ImageFont.html  # PIL.ImageFont.ImageFont
        text_to_show = "The quick brown fox jumps over the lazy dog"
        # Convert the image to RGB (OpenCV uses BGR)
        cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Pass the image to PIL
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        font_size_white_label = 15
        # use a truetype font
        font_pil = ImageFont.truetype("arial.ttf", int(font_size_white_label * factor_size_labels))
        # Draw the text
        draw.text((txt_pos_wh_x, txt_pos_wh_y), label_white_number,  fill ="black", font=font_pil)
        # Get back the image to OpenCV
        img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


        """ 5.2 generate qr code fot top right hand corner """
        # data_qr = "https://www.google.com"
        # imgage = qrcode.make(data_qr)


        """YOLOv3 text label white"""
        """ 6. define YOLO BOXES AND TEXT position"""
        ## YOLOv3 Label for white label box depends on size of white box
        YOLOv3_wht_pxl_l_t_x = wht_pxl_l_t_x - linestrength_YOLO_label
        YOLOv3_wht_pxl_l_t_y = wht_pxl_l_t_y - linestrength_YOLO_label
        YOLOv3_wht_pxl_r_b_x = YOLOv3_wht_pxl_l_t_x + wht_label_width + linestrength_YOLO_label
        YOLOv3_wht_pxl_r_b_y = YOLOv3_wht_pxl_l_t_y + wht_label_height + linestrength_YOLO_label
        # naming box
        YOLOv3_wht_pxl_l_t_x_name = YOLOv3_wht_pxl_l_t_x
        YOLOv3_wht_pxl_l_t_y_name = YOLOv3_wht_pxl_l_t_y - heigth_naming_YOLO_label
        YOLOv3_wht_pxl_r_b_x_name = YOLOv3_wht_pxl_r_b_x
        YOLOv3_wht_pxl_r_b_y_name = YOLOv3_wht_pxl_l_t_y
        # colour YOLO white label
        R_value_YOLOv3_wht_label = random.randint(0, 255)
        G_value_YOLOv3_wht_label = random.randint(0, 255)
        B_value_YOLOv3_wht_label = random.randint(0, 255)
        # VDA label
        txt_yolo_wht_dstnc_x = int( 3 * factor_size_labels)
        txt_yolo_wht_dstnc_y = int( 3 * factor_size_labels)
        txt_pos_wht_yolov3_label_x = YOLOv3_wht_pxl_l_t_x + txt_yolo_wht_dstnc_x
        txt_pos_wht_yolov3_label_y = YOLOv3_wht_pxl_l_t_y - txt_yolo_wht_dstnc_y

        """ 7. draw YOLO bounding boxes"""
        # YOLOv3 bounding box white Label
        img = cv2.rectangle(img, (YOLOv3_wht_pxl_l_t_x, YOLOv3_wht_pxl_l_t_y),
                            (YOLOv3_wht_pxl_r_b_x, YOLOv3_wht_pxl_r_b_y),
                            (B_value_YOLOv3_wht_label, G_value_YOLOv3_wht_label, R_value_YOLOv3_wht_label),
                            thickness=linestrength_YOLO_label)
        # naming box
        img = cv2.rectangle(img, (YOLOv3_wht_pxl_l_t_x_name, YOLOv3_wht_pxl_l_t_y_name),
                            (YOLOv3_wht_pxl_r_b_x_name, YOLOv3_wht_pxl_r_b_y_name),
                            (B_value_YOLOv3_wht_label, G_value_YOLOv3_wht_label, R_value_YOLOv3_wht_label), -1)

        """ 8. Write classname into boxes"""
        # # white YOLO label
        # cv2.putText(img, "white_label", (txt_pos_wht_yolov3_label_x, txt_pos_wht_yolov3_label_y), font,
        #             0.5 * factor_size_labels, (0, 0, 0),
        #             2, cv2.LINE_AA)


        ## pillow
        # Convert the image to RGB (OpenCV uses BGR)
        cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Pass the image to PIL
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        size_font_yolo_label_white = 15
        # use a truetype font and size
        font_pil = ImageFont.truetype("arial.ttf", int(size_font_yolo_label_white * factor_size_labels))
        # Draw the text
        draw.text((txt_pos_wht_yolov3_label_x, txt_pos_wht_yolov3_label_y - factor_size_labels * size_font_yolo_label_white),
                  class_name_white_VDA_9449, fill="black", font=font_pil)
        # Get back the image to OpenCV
        img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)




        """ SAVING IMAGE AND TEXT CORRECTLY for YOLOv3/v4 (11/2020) and SPLIT DATSET for CROSS-Validation"""
        """9. SAVE FILES CORRECTLY (to hand over to YOLO for custom training)"""
        #1st: Set current directory to folder underneath path of scripts to save all dataset images and textfiles"""
        dataset_path = os.getcwd() + '\complete_dataset'    ## datafolder full created dataset for YOLOv3 / YOLOv4
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        #### Todo write  more variables in string in filename for picture name and opencv operation
        #""" show and write image """
        # print(img)

        filename_str = '%s --- %s.jpg' % (label_yellow_text_combined, label_white_str)
        filename_txt = '%s --- %s.txt' % (label_yellow_text_combined, label_white_str)
        print(filename_str)
        cv2.imwrite(dataset_path + '\ ' + filename_str, img)  #creates image of img with filename via OPENCV
        # cv2.imshow(filename_str, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # convert OpenCV format to matplotlib format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plot images with VDA labels as names
        plt.imshow(img)
        plt.title(filename_str)
        #plt.savefig(filename_str)
        plt.show()  # display it

        # create textfiles with correct format for YOLO network
        # All pixel values are normalized
        # class # pixel top left corne  r x bounding box # pixel top left corner y bounding box # width of box # height box

        """10 . normalizing pixel values yellow label"""
        normalized_YOLOv3_pxl_l_t_x = float(YOLOv3_pxl_l_t_x / img.shape[1])
        normalized_YOLOv3_pxl_l_t_y = float(YOLOv3_pxl_l_t_y / img.shape[0])
        normalized_width_YOLO_label = float((yl_label_width + linestrength_YOLO_label) / img.shape[1])
        normalized_heigth_YOLO_label = float((yl_label_height + linestrength_YOLO_label) / img.shape[0])
        YOLOv3_label_string_yellow = str(class_number_yellow_VDA) + " " + \
                              str(normalized_YOLOv3_pxl_l_t_x) + " " + \
                              str(normalized_YOLOv3_pxl_l_t_y) + " " + \
                              str(normalized_width_YOLO_label) + " " + \
                              str(normalized_heigth_YOLO_label)

        """normalizing pixel values white label"""
        normalized_YOLOv3_white_pxl_l_t_x = float(YOLOv3_wht_pxl_l_t_x / img.shape[1])
        normalized_YOLOv3_white_pxl_l_t_y = float(YOLOv3_wht_pxl_l_t_y / img.shape[0])
        normalized_width_white_YOLO_label = float((wht_label_width + linestrength_YOLO_label) / img.shape[1])
        normalized_heigth_white_YOLO_label = float((wht_label_height + linestrength_YOLO_label) / img.shape[0])
        YOLOv3_label_string_white = str(class_number_white_VDA) + " " + \
                                     str(normalized_YOLOv3_white_pxl_l_t_x) + " " + \
                                     str(normalized_YOLOv3_white_pxl_l_t_y) + " " + \
                                     str(normalized_width_white_YOLO_label) + " " + \
                                     str(normalized_heigth_white_YOLO_label)

        """ write classes.names textfile"""
        #os.chdir(dataset_path)
        with open(dataset_path + '\ ' + filename_txt, 'w') as f:
            f.write(YOLOv3_label_string_yellow)
            f.write("\n")
            f.write(YOLOv3_label_string_white)



        complete_dataset.append([filename_str, filename_txt])
        complete_dataset_np = np.append(filename_str,filename_txt)

        count += 1




"""11. create classes.txt each class to detect per line"""
with open(dataset_path + '\classes.txt', 'w') as f:
    for classname in range(len(class_names)):
        f.write(class_names[classname])
        f.write("\n")


"""For GOOGLE DRIVE"""
#path_google_drive_yolov4 =

with open(dataset_path + '\obj.names', 'w') as f:
    for classname in range(len(class_names)):
        f.write(class_names[classname])
        f.write("\n")

"""create obj.data file each class to detect per line"""
with open(dataset_path + '\obj.data', 'w') as f:
    f.write("classes = {}".format(len(class_names)))
    f.write("\n")
    f.write("train = data/train.txt")
    f.write("\n")
    f.write("valid = data/test.txt")
    f.write("\n")
    f.write("names = data/obj.names")
    f.write("\n")
    f.write("backup = /mydrive/yolov{}/backup".format(yolo_version))


""""DATASET SPLIT (5 different Training-Test sets for cross-validation)"""
print("Complete dataset is:{}".format(complete_dataset))
print("Complete dataset is:{}".format(complete_dataset_np))
del complete_dataset[0]
print(complete_dataset)

total_number_images = len(complete_dataset)
print("Total number of images in dataset: {}".format(total_number_images))
random.shuffle(complete_dataset)
print("Shuffled dataset: {}".format(complete_dataset))
number_splitted_data = int(total_number_images / (number_datasets -1) )
print(number_splitted_data)

def divide_chunks(list_data, size_chunk_data):
    # loop till length list
    for number in range(0, len(list_data), size_chunk_data):
        yield complete_dataset[number:number + number_splitted_data]


chunked_list_complete_dataset = list(divide_chunks(complete_dataset, number_splitted_data))
chunked_list_complete_dataset_copy = list(divide_chunks(complete_dataset, number_splitted_data))
print(chunked_list_complete_dataset)
print(len(chunked_list_complete_dataset))

training_dataset_prt_1      = {}
training_dataset_prt_2      = {}
test_dataset                = {}
training_dataset = {}

# for k in range(len(chunked_list_complete_dataset)):
#     #test_set = chunked_list_complete_dataset.pop([k])
#     test_dataset[k] = chunked_list_complete_dataset[k]

#for nmbr_of_chunk in chunked_list_complete_dataset:

for f in range(len(chunked_list_complete_dataset)):
    print(len(chunked_list_complete_dataset))
    ## first testdatayet is last chunk and with every loop moves back to the first chunk
    if f == 0:
        dataset_test = chunked_list_complete_dataset[len(chunked_list_complete_dataset)-1 - f]
        dataset_train = chunked_list_complete_dataset[f:len(chunked_list_complete_dataset)-1]
    else:
        dataset_test = chunked_list_complete_dataset[len(chunked_list_complete_dataset)-1 - f]
        dataset_train = chunked_list_complete_dataset[:len(chunked_list_complete_dataset) -1- f ]  + chunked_list_complete_dataset[len(chunked_list_complete_dataset)-1 -f + 1:]
    test_dataset[f] = dataset_test
    training_dataset[f] = dataset_train

    print(f)
    print(training_dataset)
    print(test_dataset)


print(training_dataset)
print(test_dataset)



""" CREATE from all Data 5 Training - Test - datasets for cross-validation an mAP calculation"""
from pathlib import WindowsPath
from pathlib import PureWindowsPath
## 1st. TRAINING DATASETS
for key, value  in training_dataset.items():

# 1st: Set current directory to folder underneath path of scripts to save all dataset images and textfiles"""
    dataset_path_training_new = os.getcwd() + r'\yolov{}\Dataset_{}\obj'.format(yolo_version, key)  ## datafolder full created dataset for YOLOv3 / YOLOv4
    backup_folder_dir_for_trained_weights = os.getcwd() + r'\yolov{}\Dataset_{}\backup'.format(yolo_version, key)
    if not os.path.exists(dataset_path_training_new):
        os.makedirs(dataset_path_training_new)
        os.makedirs(backup_folder_dir_for_trained_weights)
    # 2nd remove old yolov4 dataset direcory if exist
    else:

        directory_clean = Path(os.getcwd() + r'\yolov{}\Dataset_{}\obj'.format(yolo_version, key))
        shutil.rmtree(directory_clean, ignore_errors=True)
        # creat new directory
        if not os.path.exists(dataset_path_training_new):
            os.makedirs(dataset_path_training_new)

    # copy classes.names, obj.data and obj.names into DATASET Folder
    #class.names
    source_path_class_names = dataset_path + r'\classes.txt'
    source_path_obj_data = dataset_path + r'\obj.data'
    source_path_obj_names = dataset_path + r'\obj.names'
    target_path_config_files = os.getcwd() + r'\yolov{}\Dataset_{}'.format(yolo_version, key)

    newPath = shutil.copy2(source_path_class_names, target_path_config_files)
    newPath = shutil.copy2(source_path_obj_data, target_path_config_files)
    newPath = shutil.copy2(source_path_obj_names, target_path_config_files)

    for value_list in value:

        # 2nd: Save Chunks of Training and Test Set
        file_list_to_copy = value_list

        for nmbr_file_duo_to_copy_in_list in range(len(file_list_to_copy)):
            file_duo_to_copy = file_list_to_copy[nmbr_file_duo_to_copy_in_list]

            for nmbr_single_file in range(len(file_duo_to_copy)):
                single_file_to_copy = file_duo_to_copy[nmbr_single_file]
                original_path = PurePath(dataset_path + r"\ {}".format(single_file_to_copy))
                target_path = PurePath(dataset_path_training_new ) # r"\{}".format(single_file_to_copy))
                newPath = shutil.copy2(original_path, target_path)


    ## zip obj folder for training in cloud
    path_train_zip = os.getcwd() + r'\yolov{}\Dataset_{}\obj'.format(yolo_version, key)
    root_dir_train_zip =  os.getcwd() + r'\yolov{}\Dataset_{}'.format(yolo_version, key)
    base_dir_train_zip = 'obj'
    shutil.make_archive(path_train_zip, 'zip', root_dir_train_zip, base_dir_train_zip)


## 2nd TEST DATASETS
for key, value  in test_dataset.items():
    # 1st: Set current directory to folder underneath path of scripts to save all dataset images and textfiles"""
    dataset_path_test_new = os.getcwd() + r'\yolov{}\Dataset_{}\test'.format(yolo_version, key)  ## datafolder full created dataset for YOLOv3 / YOLOv4
    if not os.path.exists(dataset_path_test_new):
        os.makedirs(dataset_path_test_new)
    # 2nd remove old yolov4 dataset directory if exist
    else:
        current_working_dir = os.getcwd()
        directory_clean = Path(current_working_dir + r'\yolov{}\Dataset_{}\test'.format(yolo_version, key))
        shutil.rmtree(directory_clean)
        # creat new directory
        if not os.path.exists(dataset_path_test_new):
            os.makedirs(dataset_path_test_new)

    for value_list in value:

        # 2nd: Save Chunks of Training and Test Set
        # file_list_to_copy = value_list
        #
        # for nmbr_file_duo_to_copy_in_list in range(len(file_list_to_copy)):
        #     file_duo_to_copy = file_list_to_copy[nmbr_file_duo_to_copy_in_list]
        file_duo_to_copy = value_list

        for nmbr_single_file in range(len(file_duo_to_copy)):
            single_file_to_copy = file_duo_to_copy[nmbr_single_file]
            original_path = PurePath(dataset_path + r"\ {}".format(single_file_to_copy))
            target_path = PurePath(dataset_path_test_new ) # r"\{}".format(single_file_to_copy))
            newPath = shutil.copy2(original_path, target_path)

    ## zip obj folder for training in cloud
    path_test_zip = os.getcwd() + r'\yolov{}\Dataset_{}\test'.format(yolo_version, key)
    root_dir_test_zip = os.getcwd() + r'\yolov{}\Dataset_{}'.format(yolo_version, key)
    base_dir_test_zip = 'test'
    shutil.make_archive(path_test_zip, 'zip', root_dir_test_zip, base_dir_test_zip)
## create backup folder inside yolov4 folder

## last step =  zip whole yolo folder  and transfer to Google drive




# for r, d, f in os.walk(path_to_training_images):
#     for file in f:
#         if file.endswith(".jpg"):
#             #print(os.path.join(r, file))
#             img_list.append(file)
#             # get absolute paths of images
#             img_list_path_absolute.append(str(path_to_training_images) + "/" + file)

""" afterwards 
===> use automatic transfer to Google Drive Python Script
1. zip all files to obj.zip folder
2 create yolov3 file in GoogleDrive 
3. drag obj.zip to yolov3 folder in GoogleDrive
"""



















#if __name__ == "__main__":
    #print_labels_on_image(img)

    # """ random label text generator"""
    # label_1st_pos = random_text_gen(2,randomascii=False, uppercase=True)
    # label_2nd_pos = random_text_gen(1, randomascii=False, uppercase=False, lowercase=False)
    # label_3rd_pos = '-'
    # label_4th_pos = random_text_gen(3, randomascii=False, uppercase=False, lowercase=False)
    # label_5th_pos = random_text_gen(1,randomascii=False, uppercase=True)
    # label_yellow_text_combined = '%s%s %s %s%s' %(label_1st_pos, label_2nd_pos, label_3rd_pos, label_4th_pos, label_5th_pos)
    # print(label_yellow_text_combined)
    # label_white_number = random_text_gen(10, randomascii=False, uppercase=False, lowercase=False)
    # print(label_white_number)
    # label_white_str = '%s' %(label_white_number)
    #
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, label_yellow_text_combined, (txt_pos_blyl_x, txt_pos_blyl_y), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    # cv2.putText(img, random_text_gen(2, randomascii=False, uppercase=False, lowercase=False), (txt_pos_ylbl_x, txt_pos_ylbl_y), font, 1, (yl_col_B,yl_col_G, yl_col_R), 2, cv2.LINE_AA)
    # cv2.putText(img, label_white_number, (txt_pos_wh_x, txt_pos_wh_y), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    #
    # #Todo write  more variables in string in filename for picture name and opencv operation
    # """ show and write image """
    # #print(img)
    # filename_str = '%s --- %s.jpg' %(label_yellow_text_combined, label_white_str)
    # print(filename_str)
    #cv2.imwrite(filename_str, img)
    # cv2.imshow('Dataset_VDA_labels', img)
    # cv2.waitKey(0) & 0xFF
    # cv2.destroyAllWindows()

#""" text generator"""
# def random_text_gen(length=32, randomascii=True, uppercase=True, lowercase=True, numbers=True):
#     character_set = ''                                                                          # lowercase, uppercase, digits etc. possile
#     if randomascii:
#         character_set += string.ascii_letters
#     elif uppercase:
#         character_set += string.ascii_uppercase
#     elif lowercase:
#         character_set += string.ascii_lowercase
#     elif numbers:
#         character_set += string.digits
#
#     return ''.join(random.choice(character_set) for i in range(length))

#"""Brown box """
# """brown box hardcoded at startpixel"""
    # pxl_l_t_x = 300                                             # pixel left top x
# pxl_l_t_y = 200
# brown_width = 600
# brown_height = 400                                          # left top y
# pxl_r_b_x = pxl_l_t_x + brown_width                         # right bottom x
# pxl_r_b_y = pxl_l_t_y + brown_height                        # right bottom y
# bb_col_R = 222
# bb_col_G = 184
# bb_col_B = 135
# """"yellow label referenced on size of top left pixel of brown box /// in range of image pixels  ###"""
# # yl_dstnc_l_t_x = 100
# # yl_dstnc_l_t_y = 40
# # yl_pxl_l_t_x = pxl_l_t_x + yl_dstnc_l_t_x                      # pixel left top
# # yl_pxl_l_t_y = pxl_l_t_y + yl_dstnc_l_t_y#

#"""diagonal blue line with thickness of 5px2"""
    # img3 = cv2.line(img3, (0,0), (511,511), (255,0,0), 5)


    ### write into image  Adding Text to Images:

    # To put texts in images, you need specify following things.
    #
    #         Text data that you want to write
    #         Position coordinates of where you want put it (i.e. bottom-left corner where data starts).
    #         Font type (Check cv2.putText() docs for supported fonts)
    #         Font Scale (specifies the size of font)
    #         regular things like color, thickness, lineType etc. For better look, lineType = cv2.LINE_AA is recommended.


# Todo yellow VDA label with YOLOv3 label
"""Yellow label  dependend on random image pixels in range of image pixels
class number 0 for YOLO labelling"""


# def yellow_label_pixels():
#     pass
#
#
# yl_pxl_l_t_x = auto_calc_start_xl_lft_tp_x  # start_pxl_lft_tp_x
# yl_pxl_l_t_y = auto_calc_start_xl_lft_tp_y  # start_pxl_lft_tp_y
# yl_pxl_r_b_x = yl_pxl_l_t_x + yl_label_width  # yellow pixel right bottom x
# yl_pxl_r_b_y = yl_pxl_l_t_y + yl_label_height  # yellow pixel right bottom y
# yl_col_R = 255
# yl_col_G = 255
# yl_col_B = int(0)
# ## black box inside yellow label depends on yellow label
# bb_pxl_l_t_x = yl_pxl_r_b_x - yl_label_height
# bb_pxl_l_t_y = yl_pxl_r_b_y - yl_label_height
# bb_pxl_r_b_x = yl_pxl_r_b_x
# bb_pxl_r_b_y = yl_pxl_r_b_y
# ## YOLOv3 Label for yellow box depends on size of yellow box
# YOLOv3_pxl_l_t_x = auto_calc_start_xl_lft_tp_x - linestrength_YOLO_label
# YOLOv3_pxl_l_t_y = auto_calc_start_xl_lft_tp_y - linestrength_YOLO_label
# YOLOv3_pxl_r_b_x = YOLOv3_pxl_l_t_x + yl_label_width + linestrength_YOLO_label
# YOLOv3_pxl_r_b_y = YOLOv3_pxl_l_t_y + yl_label_height + linestrength_YOLO_label
# # naming box
# YOLOv3_pxl_l_t_x_name = YOLOv3_pxl_l_t_x
# YOLOv3_pxl_l_t_y_name = YOLOv3_pxl_l_t_y - heigth_naming_YOLO_label
# YOLOv3_pxl_r_b_x_name = YOLOv3_pxl_r_b_x - yl_label_height
# YOLOv3_pxl_r_b_y_name = YOLOv3_pxl_l_t_y
# # colour YOLO yellow label
# R_value_YOLOv3_label = random.randint(0, 255)
# G_value_YOLOv3_label = random.randint(0, 255)
# B_value_YOLOv3_label = random.randint(0, 255)
#
# # draw Yellow VDA label box
# img = cv2.rectangle(img, (int(float(yl_pxl_l_t_x)), int(float((yl_pxl_l_t_y)))),
#                     (int(float(yl_pxl_r_b_x)), int(float(yl_pxl_r_b_y))),
#                     (yl_col_B, yl_col_G, yl_col_R), -1)
#
# # img = cv2.rectangle(img, (yl_pxl_l_t_x , yl_pxl_l_t_y), (yl_pxl_r_b_x, yl_pxl_r_b_y), (yl_col_B, yl_col_G,yl_col_R), -1)
# # img = cv2.rectangle(img, (bb_pxl_l_t_x , bb_pxl_l_t_y), (bb_pxl_r_b_x, bb_pxl_r_b_y), (0, 0, 0), -1)
#
# """CLASS 1"""
# """white label dependent on dependent/ independent start pixel
# class number 1 for YOLO labelling """
#
#
# # def draw_yellow_label():
# #     yl_pxl_l_t_x = auto_calc_start_xl_lft_tp_x  # start_pxl_lft_tp_x
# #     yl_pxl_l_t_y = auto_calc_start_xl_lft_tp_y  # start_pxl_lft_tp_y
# #     yl_pxl_r_b_x = yl_pxl_l_t_x + yl_label_width  # yellow pixel right bottom x
# #     yl_pxl_r_b_y = yl_pxl_l_t_y + yl_label_height  # yellow pixel right bottom y
# #     yl_col_R = 255
# #     yl_col_G = 255
# #     yl_col_B = int(0)
# #     ## black box inside yellow label depends on yellow label
# #     bb_pxl_l_t_x = yl_pxl_r_b_x - yl_label_height
# #     bb_pxl_l_t_y = yl_pxl_r_b_y - yl_label_height
# #     bb_pxl_r_b_x = yl_pxl_r_b_x
# #     bb_pxl_r_b_y = yl_pxl_r_b_y
# #     ## YOLOv3 Label for yellow box depends on size of yellow box
# #     YOLOv3_pxl_l_t_x = auto_calc_start_xl_lft_tp_x - linestrength_YOLO_label
# #     YOLOv3_pxl_l_t_y = auto_calc_start_xl_lft_tp_y - linestrength_YOLO_label
# #     YOLOv3_pxl_r_b_x = YOLOv3_pxl_l_t_x + yl_label_width + linestrength_YOLO_label
# #     YOLOv3_pxl_r_b_y = YOLOv3_pxl_l_t_y + yl_label_height + linestrength_YOLO_label
# #     # naming box
# #     YOLOv3_pxl_l_t_x_name = YOLOv3_pxl_l_t_x
# #     YOLOv3_pxl_l_t_y_name = YOLOv3_pxl_l_t_y - heigth_naming_YOLO_label
# #     YOLOv3_pxl_r_b_x_name = YOLOv3_pxl_r_b_x - yl_label_height
# #     YOLOv3_pxl_r_b_y_name = YOLOv3_pxl_l_t_y
# #     # colour YOLO yellow label
# #     R_value_YOLOv3_label = random.randint(0, 255)
# #     G_value_YOLOv3_label = random.randint(0, 255)
# #     B_value_YOLOv3_label = random.randint(0, 255)
#
#
# def white_label_pixels():
#     pass
#
#
# ## white  label referenced on random start pixel size of top left pixel
# """dependent / independet  of yellow label"""
# if white_depend_on_yellow_label is True:
#     wht_pxl_l_t_x = auto_calc_start_xl_lft_tp_x + wht_dstnc_l_t_x  # pixel left top white box
#     wht_pxl_l_t_y = auto_calc_start_xl_lft_tp_y + wht_dstnc_l_t_y
#     wht_pxl_r_b_x = wht_pxl_l_t_x + wht_label_width
#     wht_pxl_r_b_y = wht_pxl_l_t_y + wht_label_height
#     wht_col_R = 255
#     wht_col_G = 255
#     wht_col_B = 255
# else:
#     wht_pxl_l_t_x = auto_calc_start_xl_lft_tp_x_white  # + wht_dstnc_l_t_x  # pixel left top white box
#     wht_pxl_l_t_y = auto_calc_start_xl_lft_tp_y_white  # + wht_dstnc_l_t_y
#     wht_pxl_r_b_x = wht_pxl_l_t_x + wht_label_width
#     wht_pxl_r_b_y = wht_pxl_l_t_y + wht_label_height
#     wht_col_R = 255
#     wht_col_G = 255
#     wht_col_B = 255
#
# ## YOLOv3 Label for white label box depends on size of white box
# YOLO_wht_pxl_l_t_x = wht_pxl_l_t_x - linestrength_YOLO_label
# YOLO_wht_pxl_l_t_y = wht_pxl_l_t_y - linestrength_YOLO_label
# YOLO_wht_pxl_r_b_x = YOLO_wht_pxl_l_t_x + wht_label_width + linestrength_YOLO_label
# YOLO_wht_pxl_r_b_y = YOLO_wht_pxl_l_t_y + wht_label_height + linestrength_YOLO_label
# # naming box
# YOLOv3_wht_pxl_l_t_x_name = YOLO_wht_pxl_l_t_x
# YOLOv3_wht_pxl_l_t_y_name = YOLO_wht_pxl_l_t_y - heigth_naming_YOLO_label
# YOLOv3_wht_pxl_r_b_x_name = YOLO_wht_pxl_r_b_x
# YOLOv3_wht_pxl_r_b_y_name = YOLO_wht_pxl_l_t_y
# # colour YOLO white label
# R_value_YOLOv3_wht_label = random.randint(0, 255)
# G_value_YOLOv3_wht_label = random.randint(0, 255)
# B_value_YOLOv3_wht_label = random.randint(0, 255)
#
# # """ DRAW VDA labels and YOLOv3 labels"""
# # def print_labels_on_image(img):
#
# # img = cv2.imread('../WC9 - 390Y.jpg')
# # img = Image.open('../WC9 - 390Y.jpg')
# # print(img.size)
# # img = np.ones((1080, 1920, 3), np.uint8)*255                                                # create image with FullHD resolution in white = np.ones * 255
# # img = cv2.rectangle(img, (pxl_l_t_x, pxl_l_t_y), (pxl_r_b_x, pxl_r_b_y), (bb_col_B, bb_col_G, bb_col_R), -1)                       # thicknes  = -1 to fill rectangle                    # draw brown box BGR!!!
#
# """VDA labels"""
# # Yellow VDA label box
# img = cv2.rectangle(img, (int(float(yl_pxl_l_t_x)), int(float((yl_pxl_l_t_y)))),
#                     (int(float(yl_pxl_r_b_x)), int(float(yl_pxl_r_b_y))),
#                     (yl_col_B, yl_col_G, yl_col_R), -1)
# # Black inside Yellow label box
# img = cv2.rectangle(img, (bb_pxl_l_t_x, bb_pxl_l_t_y), (bb_pxl_r_b_x, bb_pxl_r_b_y),
#                     (0, 0, 0), -1)
# # White VDA label
# img = cv2.rectangle(img, (wht_pxl_l_t_x, wht_pxl_l_t_y), (wht_pxl_r_b_x, wht_pxl_r_b_y),
#                     (wht_col_B, wht_col_G, wht_col_R), -1)
#
# """YOLOV§ LABELS"""
# # YOLOv3 bounding box Yellow Label
# img = cv2.rectangle(img, (YOLOv3_pxl_l_t_x, YOLOv3_pxl_l_t_y), (YOLOv3_pxl_r_b_x, YOLOv3_pxl_r_b_y),
#                     (B_value_YOLOv3_label, G_value_YOLOv3_label, R_value_YOLOv3_label),
#                     thickness=linestrength_YOLO_label)
# # naming box
# img = cv2.rectangle(img, (YOLOv3_pxl_l_t_x_name, YOLOv3_pxl_l_t_y_name), (YOLOv3_pxl_r_b_x_name, YOLOv3_pxl_r_b_y_name),
#                     (B_value_YOLOv3_label, G_value_YOLOv3_label, R_value_YOLOv3_label), -1)
#
# # YOLOv3 bounding box white Label
# img = cv2.rectangle(img, (YOLO_wht_pxl_l_t_x, YOLO_wht_pxl_l_t_y), (YOLO_wht_pxl_r_b_x, YOLO_wht_pxl_r_b_y),
#                     (B_value_YOLOv3_wht_label, G_value_YOLOv3_wht_label, R_value_YOLOv3_wht_label),
#                     thickness=linestrength_YOLO_label)
# # naming box
# img = cv2.rectangle(img, (YOLOv3_wht_pxl_l_t_x_name, YOLOv3_wht_pxl_l_t_y_name),
#                     (YOLOv3_wht_pxl_r_b_x_name, YOLOv3_wht_pxl_r_b_y_name),
#                     (B_value_YOLOv3_wht_label, G_value_YOLOv3_wht_label, R_value_YOLOv3_wht_label), -1)
#
# # """ TEXT"""
# # """ random label text generator"""
# #  yellow label
# label_1st_pos = random_text_gen(2, randomascii=False, uppercase=True)
# label_2nd_pos = random_text_gen(1, randomascii=False, uppercase=False, lowercase=False)
# label_3rd_pos = '-'
# label_4th_pos = random_text_gen(3, randomascii=False, uppercase=False, lowercase=False)
# label_5th_pos = random_text_gen(1, randomascii=False, uppercase=True)
# label_yellow_text_combined = '%s%s %s %s%s' % (label_1st_pos, label_2nd_pos, label_3rd_pos, label_4th_pos, label_5th_pos)
# print(label_yellow_text_combined)
#
# # white label
# label_white_number = random_text_gen(10, randomascii=False, uppercase=False, lowercase=False)
# print(label_white_number)
# label_white_str = '%s' % (label_white_number)
#
# # """ Labels text positions"""
# # text position inside yellow label
# txt_distance_x = int(40 * factor_size_labels)
# txt_distance_y = int(5 * factor_size_labels)
# txt_pos_blyl_x = yl_pxl_l_t_x + txt_distance_x  # pixel left bottom x
# txt_pos_blyl_y = yl_pxl_l_t_y + yl_label_height - txt_distance_y  # pixel left bottom y
# # on black box
# txt_distance_yellow_x = int(6 * factor_size_labels)
# txt_distance_yellow_y = int(12 * factor_size_labels)
# txt_pos_ylbl_x = bb_pxl_l_t_x + txt_distance_yellow_x
# txt_pos_ylbl_y = bb_pxl_l_t_y + yl_label_height - txt_distance_yellow_y
# # on white label
# txt_wh_dstnc_x = int(20 * factor_size_labels)
# txt_wh_dstnc_y = int(60 * factor_size_labels)
# txt_pos_wh_x = wht_pxl_l_t_x + txt_wh_dstnc_x
# txt_pos_wh_y = wht_pxl_l_t_y + txt_wh_dstnc_y
# # on YOLOv3 label yellow VDA label
# txt_yolo_yl_dstnc_x = int(3 * factor_size_labels)
# txt_yolo_yl_dstnc_y = int(3 * factor_size_labels)
# txt_pos_yl_yolov3_label_x = YOLOv3_pxl_l_t_x + txt_yolo_yl_dstnc_x
# txt_pos_yl_yolov3_label_y = YOLOv3_pxl_l_t_y - txt_yolo_yl_dstnc_y
# # on YOLOv3 label white
# # VDA label
# txt_yolo_wht_dstnc_x = int(3 * factor_size_labels)
# txt_yolo_wht_dstnc_y = int(3 * factor_size_labels)
# txt_pos_wht_yolov3_label_x = YOLO_wht_pxl_l_t_x + txt_yolo_wht_dstnc_x
# txt_pos_wht_yolov3_label_y = YOLO_wht_pxl_l_t_y - txt_yolo_wht_dstnc_y
#
# # """Write text into boxes"""
# font = cv2.FONT_HERSHEY_SIMPLEX
# # yellow label
# cv2.putText(img, label_yellow_text_combined, (txt_pos_blyl_x, txt_pos_blyl_y), font, 0.5 * factor_size_labels, (0, 0, 0), 2,
#             cv2.LINE_AA)
# # black number
# cv2.putText(img, random_text_gen(2, randomascii=False, uppercase=False, lowercase=False),
#             (txt_pos_ylbl_x, txt_pos_ylbl_y), font, 1 * factor_size_labels, (yl_col_B, yl_col_G, yl_col_R), 2,
#             cv2.LINE_AA)
# # white label
# cv2.putText(img, label_white_number, (txt_pos_wh_x, txt_pos_wh_y), font, 0.8 * factor_size_labels, (0, 0, 0), 2,
#             cv2.LINE_AA)
#
# # yellow YOLO label
# cv2.putText(img, "yellow_label", (txt_pos_yl_yolov3_label_x, txt_pos_yl_yolov3_label_y), font, 0.5 * factor_size_labels,
#             (0, 0, 0), 2, cv2.LINE_AA)
# # white YOLO label
# cv2.putText(img, "white_label", (txt_pos_wht_yolov3_label_x, txt_pos_wht_yolov3_label_y), font,
#             0.5 * factor_size_labels, (0, 0, 0),
#             2, cv2.LINE_AA)