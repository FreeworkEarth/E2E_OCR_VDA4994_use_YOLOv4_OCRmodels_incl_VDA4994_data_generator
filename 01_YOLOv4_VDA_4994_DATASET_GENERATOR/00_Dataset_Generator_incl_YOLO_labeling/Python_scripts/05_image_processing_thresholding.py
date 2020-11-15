# Todo yellow VDA label with YOLOv3 label
"""Yellow label  dependend on random image pixels in range of image pixels
class number 0 for YOLO labelling"""


def yellow_label_pixels():
    pass


yl_pxl_l_t_x = auto_calc_start_xl_lft_tp_x  # start_pxl_lft_tp_x
yl_pxl_l_t_y = auto_calc_start_xl_lft_tp_y  # start_pxl_lft_tp_y
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
## YOLOv3 Label for yellow box depends on size of yellow box
YOLOv3_pxl_l_t_x = auto_calc_start_xl_lft_tp_x - linestrength_YOLOv3_label
YOLOv3_pxl_l_t_y = auto_calc_start_xl_lft_tp_y - linestrength_YOLOv3_label
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

# draw Yellow VDA label box
img = cv2.rectangle(img, (int(float(yl_pxl_l_t_x)), int(float((yl_pxl_l_t_y)))),
                    (int(float(yl_pxl_r_b_x)), int(float(yl_pxl_r_b_y))),
                    (yl_col_B, yl_col_G, yl_col_R), -1)

# img = cv2.rectangle(img, (yl_pxl_l_t_x , yl_pxl_l_t_y), (yl_pxl_r_b_x, yl_pxl_r_b_y), (yl_col_B, yl_col_G,yl_col_R), -1)
# img = cv2.rectangle(img, (bb_pxl_l_t_x , bb_pxl_l_t_y), (bb_pxl_r_b_x, bb_pxl_r_b_y), (0, 0, 0), -1)

"""CLASS 1"""
"""white label dependent on dependent/ independent start pixel
class number 1 for YOLO labelling """


# def draw_yellow_label():
#     yl_pxl_l_t_x = auto_calc_start_xl_lft_tp_x  # start_pxl_lft_tp_x
#     yl_pxl_l_t_y = auto_calc_start_xl_lft_tp_y  # start_pxl_lft_tp_y
#     yl_pxl_r_b_x = yl_pxl_l_t_x + yl_label_width  # yellow pixel right bottom x
#     yl_pxl_r_b_y = yl_pxl_l_t_y + yl_label_height  # yellow pixel right bottom y
#     yl_col_R = 255
#     yl_col_G = 255
#     yl_col_B = int(0)
#     ## black box inside yellow label depends on yellow label
#     bb_pxl_l_t_x = yl_pxl_r_b_x - yl_label_height
#     bb_pxl_l_t_y = yl_pxl_r_b_y - yl_label_height
#     bb_pxl_r_b_x = yl_pxl_r_b_x
#     bb_pxl_r_b_y = yl_pxl_r_b_y
#     ## YOLOv3 Label for yellow box depends on size of yellow box
#     YOLOv3_pxl_l_t_x = auto_calc_start_xl_lft_tp_x - linestrength_YOLOv3_label
#     YOLOv3_pxl_l_t_y = auto_calc_start_xl_lft_tp_y - linestrength_YOLOv3_label
#     YOLOv3_pxl_r_b_x = YOLOv3_pxl_l_t_x + yl_label_width + linestrength_YOLOv3_label
#     YOLOv3_pxl_r_b_y = YOLOv3_pxl_l_t_y + yl_label_height + linestrength_YOLOv3_label
#     # naming box
#     YOLOv3_pxl_l_t_x_name = YOLOv3_pxl_l_t_x
#     YOLOv3_pxl_l_t_y_name = YOLOv3_pxl_l_t_y - heigth_naming_YOLOv3_label
#     YOLOv3_pxl_r_b_x_name = YOLOv3_pxl_r_b_x - yl_label_height
#     YOLOv3_pxl_r_b_y_name = YOLOv3_pxl_l_t_y
#     # colour YOLO yellow label
#     R_value_YOLOv3_label = random.randint(0, 255)
#     G_value_YOLOv3_label = random.randint(0, 255)
#     B_value_YOLOv3_label = random.randint(0, 255)


def white_label_pixels():
    pass


## white  label referenced on random start pixel size of top left pixel
"""dependent / independet  of yellow label"""
if white_depend_on_yellow_label is True:
    wht_pxl_l_t_x = auto_calc_start_xl_lft_tp_x + wht_dstnc_l_t_x  # pixel left top white box
    wht_pxl_l_t_y = auto_calc_start_xl_lft_tp_y + wht_dstnc_l_t_y
    wht_pxl_r_b_x = wht_pxl_l_t_x + wht_label_width
    wht_pxl_r_b_y = wht_pxl_l_t_y + wht_label_height
    wht_col_R = 255
    wht_col_G = 255
    wht_col_B = 255
else:
    wht_pxl_l_t_x = auto_calc_start_xl_lft_tp_x_white  # + wht_dstnc_l_t_x  # pixel left top white box
    wht_pxl_l_t_y = auto_calc_start_xl_lft_tp_y_white  # + wht_dstnc_l_t_y
    wht_pxl_r_b_x = wht_pxl_l_t_x + wht_label_width
    wht_pxl_r_b_y = wht_pxl_l_t_y + wht_label_height
    wht_col_R = 255
    wht_col_G = 255
    wht_col_B = 255

## YOLOv3 Label for white label box depends on size of white box
YOLOv3_wht_pxl_l_t_x = wht_pxl_l_t_x - linestrength_YOLOv3_label
YOLOv3_wht_pxl_l_t_y = wht_pxl_l_t_y - linestrength_YOLOv3_label
YOLOv3_wht_pxl_r_b_x = YOLOv3_wht_pxl_l_t_x + wht_label_width + linestrength_YOLOv3_label
YOLOv3_wht_pxl_r_b_y = YOLOv3_wht_pxl_l_t_y + wht_label_height + linestrength_YOLOv3_label
# naming box
YOLOv3_wht_pxl_l_t_x_name = YOLOv3_wht_pxl_l_t_x
YOLOv3_wht_pxl_l_t_y_name = YOLOv3_wht_pxl_l_t_y - heigth_naming_YOLOv3_label
YOLOv3_wht_pxl_r_b_x_name = YOLOv3_wht_pxl_r_b_x
YOLOv3_wht_pxl_r_b_y_name = YOLOv3_wht_pxl_l_t_y
# colour YOLO white label
R_value_YOLOv3_wht_label = random.randint(0, 255)
G_value_YOLOv3_wht_label = random.randint(0, 255)
B_value_YOLOv3_wht_label = random.randint(0, 255)

# """ DRAW VDA labels and YOLOv3 labels"""
# def print_labels_on_image(img):

# img = cv2.imread('../WC9 - 390Y.jpg')
# img = Image.open('../WC9 - 390Y.jpg')
# print(img.size)
# img = np.ones((1080, 1920, 3), np.uint8)*255                                                # create image with FullHD resolution in white = np.ones * 255
# img = cv2.rectangle(img, (pxl_l_t_x, pxl_l_t_y), (pxl_r_b_x, pxl_r_b_y), (bb_col_B, bb_col_G, bb_col_R), -1)                       # thicknes  = -1 to fill rectangle                    # draw brown box BGR!!!

"""VDA labels"""
# Yellow VDA label box
img = cv2.rectangle(img, (int(float(yl_pxl_l_t_x)), int(float((yl_pxl_l_t_y)))),
                    (int(float(yl_pxl_r_b_x)), int(float(yl_pxl_r_b_y))),
                    (yl_col_B, yl_col_G, yl_col_R), -1)
# Black inside Yellow label box
img = cv2.rectangle(img, (bb_pxl_l_t_x, bb_pxl_l_t_y), (bb_pxl_r_b_x, bb_pxl_r_b_y),
                    (0, 0, 0), -1)
# White VDA label
img = cv2.rectangle(img, (wht_pxl_l_t_x, wht_pxl_l_t_y), (wht_pxl_r_b_x, wht_pxl_r_b_y),
                    (wht_col_B, wht_col_G, wht_col_R), -1)

"""YOLOV§ LABELS"""
# YOLOv3 bounding box Yellow Label
img = cv2.rectangle(img, (YOLOv3_pxl_l_t_x, YOLOv3_pxl_l_t_y), (YOLOv3_pxl_r_b_x, YOLOv3_pxl_r_b_y),
                    (B_value_YOLOv3_label, G_value_YOLOv3_label, R_value_YOLOv3_label),
                    thickness=linestrength_YOLOv3_label)
# naming box
img = cv2.rectangle(img, (YOLOv3_pxl_l_t_x_name, YOLOv3_pxl_l_t_y_name), (YOLOv3_pxl_r_b_x_name, YOLOv3_pxl_r_b_y_name),
                    (B_value_YOLOv3_label, G_value_YOLOv3_label, R_value_YOLOv3_label), -1)

# YOLOv3 bounding box white Label
img = cv2.rectangle(img, (YOLOv3_wht_pxl_l_t_x, YOLOv3_wht_pxl_l_t_y), (YOLOv3_wht_pxl_r_b_x, YOLOv3_wht_pxl_r_b_y),
                    (B_value_YOLOv3_wht_label, G_value_YOLOv3_wht_label, R_value_YOLOv3_wht_label),
                    thickness=linestrength_YOLOv3_label)
# naming box
img = cv2.rectangle(img, (YOLOv3_wht_pxl_l_t_x_name, YOLOv3_wht_pxl_l_t_y_name),
                    (YOLOv3_wht_pxl_r_b_x_name, YOLOv3_wht_pxl_r_b_y_name),
                    (B_value_YOLOv3_wht_label, G_value_YOLOv3_wht_label, R_value_YOLOv3_wht_label), -1)

# """ TEXT"""
# """ random label text generator"""
#  yellow label
label_1st_pos = random_text_gen(2, randomascii=False, uppercase=True)
label_2nd_pos = random_text_gen(1, randomascii=False, uppercase=False, lowercase=False)
label_3rd_pos = '-'
label_4th_pos = random_text_gen(3, randomascii=False, uppercase=False, lowercase=False)
label_5th_pos = random_text_gen(1, randomascii=False, uppercase=True)
label_text_comb = '%s%s %s %s%s' % (label_1st_pos, label_2nd_pos, label_3rd_pos, label_4th_pos, label_5th_pos)
print(label_text_comb)

# white label
label_white_number = random_text_gen(10, randomascii=False, uppercase=False, lowercase=False)
print(label_white_number)
label_white_str = '%s' % (label_white_number)

# """ Labels text positions"""
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
# on white label
txt_wh_dstnc_x = int(20 * factor_size_labels)
txt_wh_dstnc_y = int(60 * factor_size_labels)
txt_pos_wh_x = wht_pxl_l_t_x + txt_wh_dstnc_x
txt_pos_wh_y = wht_pxl_l_t_y + txt_wh_dstnc_y
# on YOLOv3 label yellow VDA label
txt_yolo_yl_dstnc_x = int(3 * factor_size_labels)
txt_yolo_yl_dstnc_y = int(3 * factor_size_labels)
txt_pos_yl_yolov3_label_x = YOLOv3_pxl_l_t_x + txt_yolo_yl_dstnc_x
txt_pos_yl_yolov3_label_y = YOLOv3_pxl_l_t_y - txt_yolo_yl_dstnc_y
# on YOLOv3 label white
# VDA label
txt_yolo_wht_dstnc_x = int(3 * factor_size_labels)
txt_yolo_wht_dstnc_y = int(3 * factor_size_labels)
txt_pos_wht_yolov3_label_x = YOLOv3_wht_pxl_l_t_x + txt_yolo_wht_dstnc_x
txt_pos_wht_yolov3_label_y = YOLOv3_wht_pxl_l_t_y - txt_yolo_wht_dstnc_y

# """Write text into boxes"""
font = cv2.FONT_HERSHEY_SIMPLEX
# yellow label
cv2.putText(img, label_text_comb, (txt_pos_blyl_x, txt_pos_blyl_y), font, 0.5 * factor_size_labels, (0, 0, 0), 2,
            cv2.LINE_AA)
# black number
cv2.putText(img, random_text_gen(2, randomascii=False, uppercase=False, lowercase=False),
            (txt_pos_ylbl_x, txt_pos_ylbl_y), font, 1 * factor_size_labels, (yl_col_B, yl_col_G, yl_col_R), 2,
            cv2.LINE_AA)
# white label
cv2.putText(img, label_white_number, (txt_pos_wh_x, txt_pos_wh_y), font, 0.8 * factor_size_labels, (0, 0, 0), 2,
            cv2.LINE_AA)

# yellow YOLO label
cv2.putText(img, "yellow_label", (txt_pos_yl_yolov3_label_x, txt_pos_yl_yolov3_label_y), font, 0.5 * factor_size_labels,
            (0, 0, 0), 2, cv2.LINE_AA)
# white YOLO label
cv2.putText(img, "white_label", (txt_pos_wht_yolov3_label_x, txt_pos_wht_yolov3_label_y), font,
            0.5 * factor_size_labels, (0, 0, 0),
            2, cv2.LINE_AA)




import pathlib
from pathlib import Path
import os
import cv2
import pytesseract

path_to_images = Path(r"C:\Users\chari\Google Drive\00_Masterthesis\Masterarbeit\00_masterthesis_code\Dataset\Repo\00_Dataset_Generator_incl_YOLO_labeling\Python_scripts\Test_data\tesseract_Bilder")

# list of images in raw image path as jpg
# just .jpg files
img_list = []
img_list_path_absolute = []
for r, d, f in os.walk(path_to_images):
    for file in f:
        if file.endswith(".jpg"):
            #print(os.path.join(r, file))
            img_list.append(file)
            # get absolute paths of images
            img_list_path_absolute.append(str(path_to_images) + "/" + file)

print(img_list)
print(len(img_list))
print(img_list[0])
print(img_list_path_absolute)



for i in range(len(img_list_path_absolute)):
    gray_image = cv2.imread(img_list[i], 0)
    tesseract_before_thresh = pytesseract.image_to_string(gray_image)
    print(tesseract_before_thresh)

    # ret, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # tesseracht_after_thresh = pytesseract.image_to_string(threshold)
    # print(tesseract_before_thresh)
    #
    # cv2.imwrite('img_to_tessseract_scan_{}.png'.format(i), threshold)
    # cv2.imshow("image_tess", threshold)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
