import glob
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage.color import rgb2gray
import os
from pathlib import Path

#Set path of data files
path = str(Path.cwd())
print("make file under",path)

if not os.path.exists( path + "/trainResized_32" ):
    os.makedirs( path + "/trainResized_32" )
if not os.path.exists( path + "/testResized_32" ):
    os.makedirs( path + "/testResized_32" )

trainFiles = glob.glob(path+"/train/*")
for i, nameFile in enumerate(trainFiles):
    image = imread(nameFile)
    imageResized = rgb2gray(resize(image,(28,28)))
    newName = path+"/trainResized_32/"+os.path.basename(nameFile)
    imsave(newName, imageResized)

testFiles = glob.glob( path + "/test/*" )
for i, nameFile in enumerate(testFiles):
    image = imread( nameFile )
    imageResized = rgb2gray(resize( image, (28,28) ))
    newName = path+"/testResized_32/"+os.path.basename(nameFile)
    imsave( newName, imageResized)