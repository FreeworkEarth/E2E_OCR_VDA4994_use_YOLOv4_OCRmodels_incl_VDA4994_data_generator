""" Helper script for VDA label drwwing methods == enables random ranking print of label """

import cv2
import numpy as np
import random
import string
import pathlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path, PurePath
from glob import glob
import glob
import os
from PIL import ImageFont, ImageDraw, Image
from random import randrange, uniform, randint
from .random_text_and_numbers_helper_function import random_text_gen

class_name_0 = "yellow_VDA_label"

labels_random_size_ratio_min = 0.2          # min 0.1
labels_random_size_ratio_max = 10         # max 10
labels_random_size_ratio_min_FullHD = labels_random_size_ratio_min / 2
labels_random_size_ratio_max_FullHD = labels_random_size_ratio_max / 2
font = cv2.FONT_HERSHEY_SIMPLEX
""" YOLOv3 Label , VDA label and picture size definition"""
## yellow VDA label:
yl_label_width_no_scaled = int(225)
yl_label_height_no_scaled  = int(50)
## white VDA label
wht_label_width_no_scaled  = int(190)
wht_label_height_no_scaled  = int(150)
#""" Define distance of white and yellow pixels ==> use in defining random start pixel area """
wht_dstnc_l_t_x_no_scaled  = int(25)
wht_dstnc_l_t_y_no_scaled  = int(100)
#"""YOLOv3 label ==> like label from https://pjreddie.com/darknet/yolo/ """
# Todo draw box around yellow and white label including name == random string inside labels
linestrength_YOLOv3_label_no_scaled  = int(2)
heigth_naming_YOLOv3_label_no_scaled  = int(25)



""" text and number generator (up to 32 signs)"""
def random_text_gen(length=32, randomascii=True, uppercase=True, lowercase=True, numbers=True):
    character_set = ''  # lowercase, uppercase, digits etc. possible
    if randomascii:
        character_set += string.ascii_letters
    elif uppercase:
        character_set += string.ascii_uppercase
    elif lowercase:
        character_set += string.ascii_lowercase
    elif numbers:
        character_set += string.digits
    return ''.join(random.choice(character_set) for i in range(length))



def draw_yellow_VDA_label(img, ):
    ## label size factors
    factor_size_labels = random.uniform(labels_random_size_ratio_min, labels_random_size_ratio_max)
    print(factor_size_labels)
    ## yellow VDA label:
    yl_label_width = int(yl_label_width_no_scaled * factor_size_labels)
    yl_label_height = int(yl_label_height_no_scaled * factor_size_labels)
    ## white VDA label
    wht_label_width = int(wht_label_width_no_scaled * factor_size_labels)
    wht_label_height = int(wht_label_height_no_scaled * factor_size_labels)
    # """ Define distance of white and yellow pixels ==> use in defining random start pixel area """
    wht_dstnc_l_t_x = int(wht_dstnc_l_t_x_no_scaled * factor_size_labels)
    wht_dstnc_l_t_y = int(wht_dstnc_l_t_y_no_scaled * factor_size_labels)
    # """YOLOv3 label ==> like label from https://pjreddie.com/darknet/yolo/ """
    # Todo draw box around yellow and white label including name == random string inside labels
    linestrength_YOLOv3_label = int(linestrength_YOLOv3_label_no_scaled * factor_size_labels)
    heigth_naming_YOLOv3_label = int(heigth_naming_YOLOv3_label_no_scaled * factor_size_labels)
    # Todo: ensure no overlappings? Ensure label including full bounding box inside generated picture
    """Start drawing labels at random places but always inside picture (including factorization)
    FOR EACH LABEL SEPARATED"""

    """Calculated by hand (Case diffenece between 4k and FullHD"""
    ##Todo: random start of yellow label ===> decide whether 4K/FullHD ===> 4k between 0 and 3840-yl_label
    # """Differences in width and height 4K"""
    # fourK_width_diff = fourK_width - yl_label_width + 2 * linestrength_YOLOv3_label
    # fourK_height_diff = fourK_height - (yl_label_height + heigth_naming_YOLOv3_label * 2 + linestrength_YOLOv3_label) #(yl_label_height + wht_dstnc_l_t_y + wht_label_height + heigth_naming_YOLOv3_label + linestrength_YOLOv3_label)
    # """Differences in width and height FullHD"""
    # FullHD_width_diff = FullHD_width - yl_label_width + 2 * linestrength_YOLOv3_label
    # FullHD_height_diff = FullHD_height - (yl_label_height + wht_dstnc_l_t_y + wht_label_height + heigth_naming_YOLOv3_label + linestrength_YOLOv3_label)
    # print(fourK_width_diff,fourK_height_diff,FullHD_width_diff, FullHD_height_diff)
    """starting pixels"""
    # start_pxl_lft_tp_x = int(float(random.uniform(0, FullHD_width_diff)))
    # start_pxl_lft_tp_y = int(float(random.uniform(0, FullHD_height_diff)))
    # print("start pixel x is: {} and start pixel in y is: {}".format(start_pxl_lft_tp_x , start_pxl_lft_tp_y))

    """Auto Calculate Difference from loaded image (WITH OpenCV ==>> height, witdth channels) from shape (=size)"""
    auto_calc_start_pxl_lft_tp_x_range_yellow = img.shape[1] - (yl_label_width + 2 * linestrength_YOLOv3_label)
    auto_calc_start_pxl_lft_tp_y_range_yellow = img.shape[0] - (
    (yl_label_height + heigth_naming_YOLOv3_label * 2 + linestrength_YOLOv3_label))
    auto_calc_start_pxl_lft_tp_x_range_white = img.shape[1] - (yl_label_width + 2 * linestrength_YOLOv3_label)
    auto_calc_start_pxl_lft_tp_y_range_white = img.shape[0] - (
    (wht_label_height + heigth_naming_YOLOv3_label * 2 + linestrength_YOLOv3_label))
    print(auto_calc_start_pxl_lft_tp_x_range_yellow, auto_calc_start_pxl_lft_tp_y_range_yellow,
          auto_calc_start_pxl_lft_tp_x_range_white, auto_calc_start_pxl_lft_tp_y_range_white)

    ## dependet on white label height
    # auto_calc_start_pxl_lft_tp_y_range_white = img.shape[0] - (yl_label_height + wht_dstnc_l_t_y + wht_label_height + heigth_naming_YOLOv3_label * 2 + linestrength_YOLOv3_label)

    # """TEST === > after in if conditional start fullHD"""
    auto_calc_start_xl_lft_tp_x_yellow = int(float(random.uniform(0, auto_calc_start_pxl_lft_tp_x_range_yellow)))
    auto_calc_start_xl_lft_tp_y_yellow = int(float(random.uniform(0, auto_calc_start_pxl_lft_tp_y_range_yellow)))
    auto_calc_start_xl_lft_tp_x_white = int(float(random.uniform(0, auto_calc_start_pxl_lft_tp_x_range_white)))
    auto_calc_start_xl_lft_tp_y_white = int(float(random.uniform(0, auto_calc_start_pxl_lft_tp_y_range_white)))
    print(auto_calc_start_xl_lft_tp_x_yellow, auto_calc_start_xl_lft_tp_y_yellow, auto_calc_start_xl_lft_tp_x_white,
          auto_calc_start_xl_lft_tp_y_white)

    """
    CLASS 0
    Yellow label  dependend on random image pixels in range of image pixels
          class number 0 for YOLO labelling
    """

    """ 1. define yellow label pixels"""
    yl_pxl_l_t_x = auto_calc_start_xl_lft_tp_x_yellow  # start_pxl_lft_tp_x
    yl_pxl_l_t_y = auto_calc_start_xl_lft_tp_y_yellow  # start_pxl_lft_tp_y
    yl_pxl_r_b_x = yl_pxl_l_t_x + yl_label_width  # yellow pixel right bottom x
    yl_pxl_r_b_y = yl_pxl_l_t_y + yl_label_height  # yellow pixel right bottom y
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
    label_text_comb = '%s%s %s %s%s' % (label_1st_pos, label_2nd_pos, label_3rd_pos, label_4th_pos, label_5th_pos)
    print(label_text_comb)

    """ 4. define Labels text positions"""
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

    """ 5. draw text for labels """
    # yellow label
    cv2.putText(img, label_text_comb, (txt_pos_blyl_x, txt_pos_blyl_y), font, 0.5 * factor_size_labels, (0, 0, 0),
                2, cv2.LINE_AA)
    # black number
    cv2.putText(img, random_text_gen(2, randomascii=False, uppercase=False, lowercase=False),
                (txt_pos_ylbl_x, txt_pos_ylbl_y), font, 1 * factor_size_labels, (yl_col_B, yl_col_G, yl_col_R), 2,
                cv2.LINE_AA)

    """YOLO text (text = classname)"""
    """ 6. define YOLO BOXES AND TEXT position"""
    ## YOLOv3 Label for yellow box depends on size of yellow box
    YOLOv3_pxl_l_t_x = auto_calc_start_xl_lft_tp_x_yellow - linestrength_YOLOv3_label
    YOLOv3_pxl_l_t_y = auto_calc_start_xl_lft_tp_y_yellow - linestrength_YOLOv3_label
    YOLOv3_pxl_r_b_x = YOLOv3_pxl_l_t_x + yl_label_width + linestrength_YOLOv3_label
    YOLOv3_pxl_r_b_y = YOLOv3_pxl_l_t_y + yl_label_height + linestrength_YOLOv3_label
    # naming box
    YOLOv3_pxl_l_t_x_name = YOLOv3_pxl_l_t_x
    YOLOv3_pxl_l_t_y_name = YOLOv3_pxl_l_t_y - heigth_naming_YOLOv3_label
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
    # on YOLOv3 label yellow VDA label
    txt_yolo_yl_dstnc_x = int(3 * factor_size_labels)
    txt_yolo_yl_dstnc_y = int(3 * factor_size_labels)
    txt_pos_yl_yolov3_label_x = YOLOv3_pxl_l_t_x + txt_yolo_yl_dstnc_x
    txt_pos_yl_yolov3_label_y = YOLOv3_pxl_l_t_y - txt_yolo_yl_dstnc_y
    """ 7. draw YOLO bounding boxes"""
    # YOLOv3 bounding box Yellow Label
    img = cv2.rectangle(img, (YOLOv3_pxl_l_t_x, YOLOv3_pxl_l_t_y), (YOLOv3_pxl_r_b_x, YOLOv3_pxl_r_b_y),
                        (B_value_YOLOv3_label, G_value_YOLOv3_label, R_value_YOLOv3_label),
                        thickness=linestrength_YOLOv3_label)
    # naming box
    img = cv2.rectangle(img, (YOLOv3_pxl_l_t_x_name, YOLOv3_pxl_l_t_y_name),
                        (YOLOv3_pxl_r_b_x_name, YOLOv3_pxl_r_b_y_name),
                        (B_value_YOLOv3_label, G_value_YOLOv3_label, R_value_YOLOv3_label), -1)

    """8. Write classname into boxes"""
    # yellow YOLO label
    cv2.putText(img, class_name_0, (txt_pos_yl_yolov3_label_x, txt_pos_yl_yolov3_label_y), font,
                0.5 * factor_size_labels, (0, 0, 0), 2, cv2.LINE_AA)

    return img