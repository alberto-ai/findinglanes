import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
import cv2

# Define directories and read images
dir_test_images = './test_images/'
dir_test_videos = './test_videos/'

# Read test images
files = [dir_test_images + f for f in listdir(dir_test_images)]

# REGION FILTER
# Define region of interest: four vertices
def get_vertices(width, height, width_top, padding_bottom, horizon):
    top_left = [width / 2 - (width_top / 2), horizon]
    top_right = [width / 2 + (width_top / 2), horizon]
    bottom_left = [padding_bottom, height]
    bottom_right = [width - padding_bottom, height]
    return np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

def mask_image_region(image):
    mask = np.zeros_like(image)
    ignore_mask_color = 255
    vertices = get_vertices(image.shape[1], image.shape[0], 20, 30, 310)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(image, mask)

# CANNY FILTER
low_threshold = 40
high_threshold = 120
def canny_filter(image, low_threshold, high_threshold):
    kernel_size = 5
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    return cv2.Canny(blur_gray, low_threshold, high_threshold)

# HOUGH TRANSFORM
# Define Hough transform parameters
rho = 2                       # distance resolution in pixels of the Hough grid
theta = np.pi/180             # angular resolution in radians of the Hough grid
threshold = 15                # minimum number of votes (intersections in Hough grid cell)
min_line_length = 30          # minimum number of pixels making up a line
max_line_gap = 20             # maximum gap in pixels between connectable line segments

def Hough_all_lines(image, gray_image, rho, theta, threshold, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(gray_image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    return lines

def points_closest_to_bottom_and_top(lines):
    left = []
    right = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y1-y2) / float((x1-x2))

            if slope >= 0:
                left.append([x1,y1])
                left.append([x2,y2])
            else:
                right.append([x1,y1])
                right.append([x2,y2])

    left_bottom_point = max(left, key = lambda t: t[1])
    right_bottom_point = max(right, key = lambda t: t[1])
    left_top_point = min(left, key = lambda t: t[1])
    right_top_point = min(right, key = lambda t: t[1])

    return [(left_bottom_point, left_top_point), (right_bottom_point, right_top_point)]

def slopes_average_lines(lines):
    left_slopes = []
    right_slopes = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y1-y2) / float((x1-x2))
            intercept = y1 - slope * x1
            if slope >= 0:
                left_slopes.append(slope)
            else:
                right_slopes.append(slope)

    left = np.asarray(left_slopes)
    right = np.asarray(right_slopes)

    return (np.mean(left), np.mean(right))

def get_points_at_top_and_bottom(slope, point, height, height_limit):
    print point
    intercept = point[1] - slope * point[0]
    bottom = ((height - intercept) / slope, height)
    top = ((height_limit-intercept) / slope, height_limit)

    return [int(round(top[0],0)), int(round(top[1],0)), int(round(bottom[0],0)), int(round(bottom[1],0))]

def extend_line(line, horizon_height, height):
    # y = mx + b
    slope = (line[0][1] - line[1][1]) / float(line[0][0] - line[1][0])
    intercept = line[0][1] - (slope * line[0][0])

    # x = (y - b) / m
    a = (int((height - intercept) / slope), height)
    b = (int((horizon_height - intercept) / slope), horizon_height)

    return [a, b]

def get_Hough_average_lines(image, lines):
    lines_image = np.copy(image)*0

    left_avg_slope, right_avg_slope = slopes_average_lines(lines)
    points = points_closest_to_bottom_and_top(lines)
    left_line = points[0]
    right_line = points[1]

    left_line = extend_line(left_line, 320, image.shape[1])
    right_line = extend_line(right_line, 320, image.shape[1])

    cv2.line(lines_image, (left_line[0][0], left_line[0][1]), (left_line[1][0], left_line[1][1]), (255,0,0), 10)
    cv2.line(lines_image, (right_line[0][0], right_line[0][1]), (right_line[1][0], right_line[1][1]), (255,0,0), 10)

    return lines_image

def process_image(image):
    current_image = np.copy(image)
    image_canny = canny_filter(current_image, low_threshold, high_threshold)
    image_canny_region = mask_image_region(image_canny)
    lines = Hough_all_lines(current_image, image_canny_region,  rho, theta, threshold, min_line_length, max_line_gap)
    lines_image_avg = get_Hough_average_lines(current_image, lines)
    image_edges = cv2.addWeighted(current_image, 0.8, lines_image_avg, 1, 0)

    return image_edges

# Tests
# for onefile in files:
#     print onefile
#     image = mpimg.imread(onefile)
#     lines_edges = process_image(image)
#     plt.imshow(lines_edges)
#     plt.show()

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

white_output = dir_test_videos + 'output/' + 'solidWhiteRight.mp4'
clip1 = VideoFileClip(dir_test_videos + 'solidWhiteRight.mp4')
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

yellow_output = dir_test_videos + 'output/' + 'solidYellowLeft.mp4'
clip2 = VideoFileClip(dir_test_videos + 'solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = dir_test_videos + 'output/' + 'challenge.mp4'
clip3 = VideoFileClip(dir_test_videos + 'challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)
