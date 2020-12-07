""" CHOOOOSE YOUUUUUR YOLO SETTTTINGS"""
JUST_TRAINING = 0  # 0 for no , 1 for yes
YOLO_major_version = "v4" #"v4"
YOLO_minor_version = 416 # defines yolo size all multiples of 32

number_of_custom_training_classes = 64 # or 1 or more
filename_dataset = "Dataset_0"
filename_cfg = "yolov4-cstm.cfg"        # configurations filename
#filename_test_cfg = "yolov4_test_custom_dataset.cfg"   # configurations filename

train = 1 ## 1 is training, 0 (else) is testing



dataset_size      = 1000        # number of images in your dataset
datasets_folder   = "yolov4"   # leave or refractor all
filename_dataset  = "obj.zip"  # leave or refractor all
foldername_trainingset = "obj"
filename_testset  = "test.zip" # leave or refractor all
foldername_testset = "test"


#### test OR train (subdivisions = 1 for test, subdivisions = 16 for test)
if train == 1:
  subdivisions = 16
else:
  subdivisions = 1

############
import os
from pathlib import Path, PosixPath, PurePath, PureWindowsPath, PurePosixPath
import shutil
import glob

# ### MOUNT Google Drive
# from google.colab import drive
# drive.mount('/content/gdrive',force_remount=True)
# # this creates a symbolic link so that now the path /content/gdrive/My\ Drive/ is equal to /mydrive
# !ln -s /content/gdrive/My\ Drive/ /mydrive
# !ls /mydrive
#
# # define helper functions
# def imShow(path):
#   import cv2
#   import matplotlib.pyplot as plt
#   %matplotlib inline
#
#   image = cv2.imread(path)
#   height, width = image.shape[:2]
#   resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)
#
#   fig = plt.gcf()
#   fig.set_size_inches(18, 10)
#   plt.axis("off")
#   plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
#   plt.show()
#
# # use this to upload files
# def upload():
#   from google.colab import files
#   uploaded = files.upload()
#   for name, data in uploaded.items():
#     with open(name, 'wb') as f:
#       f.write(data)
#       print ('saved file', name)
#
# # use this to download a file
# def download(path):
#   from google.colab import files
#   files.download(path)
#
#   ## copy dataset from Google Drive folder into contents folder of Colab
#   if dataset_size < 500:
#       from google.colab import files
#       uploaded = files.upload()
#   else:
#       print("Dataset to big for automatic upload \n ====> Upload your zipfile manually into main GoogleDrive Folder")
#       shutil.copy2('/mydrive/yolov4/{}'.format(filename_dataset), '/content/{}'.format(filename_dataset))
#       shutil.copy2('/mydrive/yolov4/{}'.format(filename_testset), '/content/{}'.format(filename_testset))



# automatic creation of helper folder yolo + version if not exists
if not os.path.exists("/mydrive/yolo" + "{}".format(YOLO_major_version)):
    os.makedirs("/mydrive/yolo" + "{}".format(YOLO_major_version))


# Check if NVIDIA GPU is enabled and which you got
!nvidia-smi


# verify CUDA
!/usr/local/cuda/bin/nvcc --version


!git clone https://github.com/AlexeyAB/darknet

# change makefile to have GPU and OPENCV enabled
%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
print("1")
!sed -i 's/GPU=0/GPU=1/' Makefile
print("2")
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
print("3")
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!make
## make Darknet


# unzip the zip file and its contents should now be in /darknet/data/obj

os.chdir("/content")
print(os.getcwd())
!unzip $filename_dataset -d /content/darknet/data/
!unzip $filename_testset -d /content/darknet/data/


os.chdir("/content/darknet")

!cp /mydrive/yolov4/generate_train.py ./
!cp /mydrive/yolov4/generate_test.py ./

!python generate_train.py
!python generate_test.py


## copy in google colab into darknet/cfg folder with new filename
os.chdir("/content/darknet")
!cp cfg/yolov4-custom.cfg cfg/$filename_cfg

## in mydrive (google drive)
## download cfg to google drive and change its name
#!cp cfg/yolov3.cfg /mydrive/yolov3/$filename     #yolov3_training.cfg

## work on cfg file

##upload and close again
# upload the custom .cfg back to cloud VM from Google Drive
#!cp /mydrive/yolov3/yolov3_custom.cfg ./cfg

# upload the custom .cfg back to cloud VM from local machine (uncomment to use)
#%cd cfg
#upload()
#%cd ..


# number_custom_training_classes = nmbr_cstm_trng_classes

## subdivisions if is in detecton mode:
##batch from 1 to 64 (or it stays same)
!sed - i.bak
"s/batch=1/batch=64/" "cfg/$filename_cfg"
## subdivisons from 1 to 16
!sed - i.bak
"s/subdivisions=1/subdivisions=16/" "cfg/$filename_cfg"

## batches
!sed - i.bak
"s/batch=1/batch=64/" "cfg/$filename_cfg"

## subdivisons
## 1 is for testing
if train == 1:
    !sed - i.bak
    "s/subdivisions=8/subdivisions=16/" "cfg/$filename_cfg"
else:
    !sed - i.bak
    "s/subdivisions=8/subdivisions=1/" "cfg/$filename_cfg"

# change with and height dependend on dataset resolution basic is 416 to 416 (w x h)
str_width = "width={}".format(YOLO_minor_version)
str_height = "height={}".format(YOLO_minor_version)

!sed - i.bak
"s/width=608/$str_width/" "cfg/$filename_cfg"
!sed - i.bak
"s/height=608/$str_height/" "cfg/$filename_cfg"

## max batches
max_batches = 2000 * number_of_custom_training_classes

if max_batches <= 6000:
    max_batches = 6000
else:
    max_batches = max_batches

string_sed_command_max_batches = 'max_batches = ' + str(max_batches)
# print(string_sed_command_max_batches)
!sed - i.bak
"s/max_batches = 500500/$string_sed_command_max_batches/" "cfg/$filename_cfg"

## step size
step_size_80_percent = int(0.8 * max_batches)
step_size_90_percent = int(0.9 * max_batches)
# print(step_size_80_percent, step_size_90_percent)
string_step_size = "steps={},{}".format(step_size_80_percent, step_size_90_percent)
# print(string_step_size)
!sed - i.bak
"s/steps=400000,450000/$string_step_size/" "cfg/$filename_cfg"


##classes
# all classes at once
!sed - i.bak
"s/classes=80/classes=$number_of_custom_training_classes/" "cfg/$filename_cfg"

# classes by specific line
# !sed -i "610 s@classes=80@classes=$number_custom_training_classes@" "$filename"
# !sed -i "696 s@classes=80@classes=$number_custom_training_classes@" "$filename"
# !sed -i "783 s@classes=80@classes=$number_custom_training_classes@" "$filename"


## Filters
number_filters = (number_of_custom_training_classes + 5) * 3
# print(number_filters)
# all filters at once
!sed - i.bak
"s/filters=255/filters=$number_filters/" "cfg/$filename_cfg"

# !sed -i '603 s@filters=255@filters=18@' cfg/yolov3_training.cfg
# !sed -i '689 s@filters=255@filters=18@' cfg/yolov3_training.cfg
# !sed -i '776 s@filters=255@filters=18@' cfg/yolov3_training.cfg


print(os.getcwd())



# obj.names = copy classes.txt (entries same as in classes.txt of labelling dataset)
print(os.getcwd())
os.chdir("/content/darknet")
#Python
shutil.copy2('/mydrive/yolov4/classes.txt', '/content/darknet/data/')
shutil.copy2('/mydrive/yolov4/obj.names', '/content/darknet/data/')

# bash
#!cp /content/darknet/data/obj/classes.txt /content/darknet/data/obj.names

#with open(dataset_path + '\obj.names', 'w') as f:
    #f.write(class_name_0)
    #f.write("\n")
    #f.write(class_name_1)



# obj.data = definition of custom training handling of files
with open('data/obj.data', 'w') as f:
    f.write("classes = {}".format(number_of_custom_training_classes))
    f.write("\n")
    f.write("train = data/train.txt")
    f.write("\n")
    f.write("valid = data/test.txt")
    f.write("\n")
    f.write("names = data/obj.names")
    f.write("\n")
    #f.write("backup = /mydrive/yolo{}/backup".format(YOLO_version))
    f.write("backup = /mydrive/yolo" + "{}/backup".format(YOLO_major_version))


# create backup folder if doesnt exists
if not os.path.exists("/mydrive/yolo" + "{}/backup".format(YOLO_major_version)):
    os.makedirs("/mydrive/yolo" + "{}/backup".format(YOLO_major_version))

#"""upload obj.names and obj.datailes manually from drive or local  folder"""
# upload the obj.names and obj.data files to cloud VM from Google Drive
#!cp /mydrive/yolov3/obj.names ./data
#!cp /mydrive/yolov3/obj.data  ./data

# upload the obj.names and obj.data files to cloud VM from local machine (uncomment to use)
#%cd data
#upload()
#%cd ..



# get yolov3 pretrained coco dataset weights
os.chdir("/content/darknet")
#!wget https://pjreddie.com/media/files/yolov3.weights
# Download weights darknet model 53
#!wget https://pjreddie.com/media/files/darknet53.conv.74

# train your custom detector
os.chdir("/content/darknet")

#!./darknet detector train data/obj.data cfg/$filename_cfg darknet53.conv.74 -dont_show

#!./darknet detector train data/obj.data cfg/yolov4_training_custom_dataset.cfg yolov4.conv.137 -dont_show #-map
#!./darknet detector train data/obj.data cfg/$filename_cfg yolov4.conv.137 -dont_show -map

!./darknet detector train data/obj.data cfg/yolov4-cstm.cfg yolov4.conv.137 -dont_show -map
#!./darknet detector train data/obj.data cfg/yolov4-obj.cfg yolov4.conv.137 -dont_show -map
#!./darknet detector train data/obj.data $filename_cfg yolov3.weights -dont_show

!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137


# if yolo stops
os.chdir("/content/darknet")
#!cp /mydrive/yolov4/yolov4-obj.cfg ./cfg
#!./darknet detector train data/obj.data cfg/$filename_cfg darknet53.conv.74 -dont_show
!./darknet detector train data/obj.data cfg/yolov4-obj.cfg /mydrive/yolov4//backup/yolov4-obj_last.weights -dont_show



# need to set our custom cfg to test mode

##batch from 64 to 1
#!sed -i.bak "s/batch=64/batch=1/" "cfg/yolov4-obj.cfg"
## subdivisons from 16 to 1
#!sed -i.bak "s/subdivisions=16/subdivisions=1/" "cfg/yolov4-obj.cfg"

##batch from 64 to 1
!sed -i.bak "s/batch=64/batch=1/" "cfg/$filename_cfg"
## subdivisons from 16 to 1
!sed -i.bak "s/subdivisions=16/subdivisions=1/" "cfg/$filename_cfg"



# run your custom detector with this command (upload an image to your google drive to test, thresh flag sets accuracy that detection must be in order to show it)

!./darknet detector test data/obj.data cfg/yolov4-obj.cfg /mydrive/yolov4/backup/yolov4-obj_best.weights /content/gdrive/MyDrive/yolov4/test/test_VDA.jpg -thresh 0.1
imShow('predictions.jpg')