import cv2 as cv
from cv2 import resize
import numpy as np
import math
import statistics
from statistics import mode



#Functions
def rescaleFrame(frame, scale = 2):
    width = int(frame.shape[1] * scale)
    heigth = int(frame.shape[0] * scale)
    dimension = (width,heigth)

    return cv.resize(frame, dimension, interpolation=cv.INTER_NEAREST)

#Function to find the most common value
def most_common(List):
    return(mode(List))


def entrance():
    image = cv.imread("../Pictures/foto2.png")

    image_copy = image.copy()

    #Upscale the original picture 3 times, this to make sure the waveguide is bigger.
    new_img = rescaleFrame(image , 3)

    #Crop the upscaled picture to a smaller surface to detect the waveguide
    #                              y        x
    new_img_cropped = new_img[3750:4800, 4950:5700]
    cv.imwrite("newpic.png", new_img_cropped)

    new_img_copy = new_img_cropped.copy()
    #cv.imshow('Zoomed and cropperd', new_img_copy)

    #Threshold and edge detection algorithme
    gray_image = cv.cvtColor(new_img_cropped, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray_image, 130, 235, cv.THRESH_BINARY)
    canny = cv.Canny(thresh, 200, 175)

    #cv.imshow('thresh', thresh)

    #Finding all the vertical lines
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 10))
    detect_vertical = cv.morphologyEx(canny, cv.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv.findContours(detect_vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # # Add debugging: Check if any contours were found
    # if not cnts:
    #     print("No vertical contours were detected.")
    #     return None, None  # Exit the function if no contours were found

    for c in cnts:
        cv.drawContours(new_img_copy, [c], -1, (36,255,12), 2)

    #Put all x location data in a new array.
    all_array = []
    all_array.extend(cnts)
    new_array = []
    #Sort all x value's to a new array called new_array, after that the highest and lowest value will be sorted.
    i = 0
    lowest_value = 9999
    highest_value = -1
    # --------------- Newly added 02/07/2024------------------------------
    outlier_threshold = 50  # pixels
    # trying to exclude the  outliers when one of the points is detected
    vertical_detections = [detection[0][0][0] for detection in all_array]
    mean_detection = statistics.mean(vertical_detections)

    filtered_detections = [detection for detection in vertical_detections if abs(detection - mean_detection) <= outlier_threshold]

    total_value = sum(filtered_detections)
    num_detections = len(filtered_detections)

    #Finding center of the detections
    vertical_line = total_value/num_detections
    # ----------------------------------------------------------------------------------------

    round_up_center_value = round(vertical_line)

    #Print the pixel location.
    Calculated_x_pixel = ((vertical_line/3)+1650)
    #print('Original X pixel:', Calculated_x_pixel)

    #Draw the line in the center
    result = cv.line(new_img_copy, (round_up_center_value, 0), (round_up_center_value,2000), (255,0,0), 2)
    #cv.imshow('result', result)
    cv.imwrite("vision_result.png", result)

    #Find horizontal line
    new_img_cropped_horizontal = image_copy[1030:2200]
    canny_horizontal = cv.Canny(new_img_cropped_horizontal, 100, 200)
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,1))
    detect_horizontal = cv.morphologyEx(canny_horizontal, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv.findContours(detect_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]


    #cv.imshow('jhkjh', new_img_cropped_horizontal)

    #Filter horizontal pixel location
    all_array_horizontal = []
    all_array_horizontal.extend(cnts)

    new_array_horizontal = []
    for x in all_array_horizontal:
        i += 1
        if i < len(all_array_horizontal):
           new_array_horizontal.append(all_array_horizontal[i][0][0][1])

    #Find common horizontal line
    Calculated_y_pixel = (most_common(new_array_horizontal))
    #print('Original Y pixel:', Calculated_y_pixel+1250)

    #Draw vertical line
    #result_original = cv.line(image_copy, (Calculated_x_pixel,0), (Calculated_x_pixel,2000), (255,0,0), 2)

    #Draw horizontal line
    #img_copy = cv.line(result_original, (0,most_common(new_array_horizontal)), (9000,most_common(new_array_horizontal)), (255,0,0), 2)


    #Downscale picture for demonstartion
    #result_original = rescaleFrame(result_original, 0.5)
    #cv.imshow('result_original', result_original)s

    cv.waitKey(0)
    return Calculated_x_pixel, Calculated_y_pixel+1250


entrance()

