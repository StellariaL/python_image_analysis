import cv2 as cv
from matplotlib import pyplot as plt
import body_axis

imgpath="D:\\Xiong Lab\\20250306-confocal\\e04-Bra-maxproj-flipped.tif"

img=body_axis.load_image(imgpath)

otsu=body_axis.otsu_threshold(img)

open_binary=body_axis.opening(otsu,11)

closed_binary=body_axis.closing(open_binary,11)

body_axis.show_images([img,closed_binary])