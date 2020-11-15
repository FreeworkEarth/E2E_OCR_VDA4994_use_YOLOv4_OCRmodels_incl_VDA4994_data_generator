#!/usr/bin/env python
# above line to execute in linux
"""ONE FILE TO LABEL THEM ALL"""
import numpy as np
import cv2
import random
import string
import pathlib
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

# """        Generalized for alphanumeric"""
# ## write classes:
#
#
#
# ### id_alphanumeric
# amount_id_alphanumeric = 35  # 0-9 and a-z = 36 alphanumeric numbers starting with 0
# ### random id
# random_id_alphanumeric = int(float(random.uniform(0,amount_id_alphanumeric)))
# print(random_id_alphanumeric)
#
# size_alphanumeric_generic = 1
# factor_size_alphanumeric = random.uniform(labels_random_size_ratio_min, labels_random_size_ratio_max)
# printsize_alphanumeric = size_alphanumeric_generic * factor_size_alphanumeric
#
#
# ### print alphanumerics
# counter_number_alphanumeric = 5
# while i < counter_number_alphanumeric:
#
#     #alphanumeric_print = alphanumeric_dict{radnom_id_alphanumeric}
#
#     ### PRINT IN TXT
#         #f.write in textfile of classes for bounding box
#     #### if random_id_alphanumeric == random_id_alphanumeric)
#
#     pass
# ### size alphanumeric





# class alphanumeric:
#
#     def __init__(self):
#         self.class_names_list_bydict = []
#         self.class_values_list_bydict = []

#def alphanumeric_random():

dict_classes_alphanumeric = {}
string_alphanumeric_numbers = "0123456789"
string_alphanumeric_lowercase = "abcdefghijklmnopqrstuvwxyz"
string_alphanumeric_uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

string_alphanumeric = string_alphanumeric_numbers + string_alphanumeric_uppercase + string_alphanumeric_lowercase
number_classes_alphanumeric = len(string_alphanumeric)
print(number_classes_alphanumeric)
print(len(string_alphanumeric))
print(string_alphanumeric)

class_names_list = []
class_values_list_string = []
class_values_list = []

# """ all through dictionary = dynamic"""
class_names_list_bydict = []
class_values_list_bydict = []
class_names = dict_classes_alphanumeric.keys()
class_values = dict_classes_alphanumeric.values()

cnt_string_alphnmbr = 0
for cnt_string_alphnmbr in range(len(string_alphanumeric)):
    dict_classes_alphanumeric[cnt_string_alphnmbr] = string_alphanumeric[cnt_string_alphnmbr]
    number_classes = len(dict_classes_alphanumeric)
    class_names_list.append("class_name_{}".format(cnt_string_alphnmbr))
    class_values_list_string.append(string_alphanumeric[cnt_string_alphnmbr])

for key, value in dict_classes_alphanumeric.items():
    class_names_list_bydict.append(key)
    # class_names_list_solo_int.append"{}"
    class_values_list_bydict.append(dict_classes_alphanumeric[key])

dict_classes_alphanumeric_tuple = dict_classes_alphanumeric.items()
print(dict_classes_alphanumeric_tuple)
print(class_names_list)
print(class_values_list)
print(class_names)
print(class_values)

######
print(class_names_list_bydict)
print(class_values_list_bydict)
print(dict_classes_alphanumeric)

""" print random alphanumeric sign into loaded picture"""
number_printed_alphanumerical = 50

nmbr_alphanum = 0
for nmbr_alphanum in range(number_printed_alphanumerical):
    random_alphanumerical_key = random.randint(0, len(class_names_list_bydict) - 1)
    # print(random_alphanumerical_key)

    value_random_alphanumerical = dict_classes_alphanumeric[random_alphanumerical_key]
    print(value_random_alphanumerical)


#"""11. create classes.txt each class to detect per line"""
dataset_path = os.getcwd() + '\obj'

with open(dataset_path + '\classes_all_alphanumeric.txt', 'w') as f:
    for j in range(len(class_names_list_bydict)):
        f.write("class_name_{}".format(class_names_list_bydict[j]))
        f.write("\n")
        #f.write(class_names[j])

with open(dataset_path + '\obj.names_all_alphanumeric', 'w') as f:
    for j in range(len(class_names_list_bydict)):
        f.write("{}".format(class_values_list_bydict[j]))
        f.write("\n")


"""create obj.data file each class to detect per line"""
with open(dataset_path + '\obj.data_alphanumeric_all', 'w') as f:
    f.write("classes = {}".format(len(class_names_list_bydict)))
    f.write("\n")
    f.write("train = data/train.txt")
    f.write("\n")
    f.write("valid = data/test.txt")
    f.write("\n")
    f.write("names = data/obj.names")
    f.write("\n")
    f.write("backup = /mydrive/yolov4/backup")



#return [value_random_alphanumerical, class_names_list_bydict, class_values_list_bydict]


#
# if __name__ == "__main__":
#     # execute only if run as a script
#     alphanumeric_random()
