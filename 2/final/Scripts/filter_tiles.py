from PIL import Image
import numpy as np
import os

#  This script finds all the images whose sum of crop pixels and weed pixels is under a chosen threshold, and removes them
#  The images to delete are written in a .txt file first, that will be retrieved later for the final deletion

#  This script exploits an already existing dataset with tiles, from which it will remove images. 
#  So you must first create a "tiled" dataset with make_tiles.py

threshold = 6000  # sum of crop and weed pixels above which we keep the image

file1_path = os.path.join('folder where you want to store and retrieve the temporary file with images to remove', "to_remove.txt")

directory = 'path of training folder'  #  this is the path of thre "training" folder, that contains the folders of the various teams




def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def isInteresting(filepath):
#  returns true if the image has to be kept

    tot = 0

    image = Image.open(filepath)
    image_array = np.array(image)

    unique, counts = np.unique(image_array, return_counts=True)
    dict1 = dict(zip(unique, counts))

    if 255 in dict1:
        tot += int(dict1[255] / 3)

    if 67 in dict1:
        tot += dict1[67]

    return tot > threshold


file1 = open(file1_path, "w")

png_paths = []
for filepath in absoluteFilePaths(directory):
    if filepath[-4:] == ".png":
        png_paths.append(filepath)

print("checking if images are above threshold...")
for filepath in png_paths:
    if not isInteresting(filepath):
        print(filepath[:-4].replace("Masks", "Images") + ".jpg", file=file1)
        print(filepath, file=file1)

file1.close()

print("done")



# now the previously written file is retrieved and images are removed

file1 = open(file1_path, "r")

lines_without_newline = file1.read().splitlines()

print("deleting files...")
for line in lines_without_newline:
    os.remove(line)
print("done")
