import cv2
import matplotlib.pyplot as plt
import numpy as np


def img_to_grayscale(img):
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_grayscale


def img_to_bin(img):
    thresh, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin, thresh


def get_contours(img):
    img, contours, hierarchy = hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours, hierarchy


def invert_image(img):
    return cv2.bitwise_not(img)


def select_roi(contours, img_width, img_height):
    good_contours = []
    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        width, height = size
        img_area = img_width * img_height
        contour_area = width * height
        if contour_area > img_area / 13:
            good_contours.append(contour)
    return good_contours


def crop(img, img_orig):
    image, contours, hierarchy = get_contours(img)
    good_contours = select_roi(contours, len(img_orig[0]), len(img_orig[1]))
    images = []
    for contour in good_contours:
        x, y, w, h = cv2.boundingRect(contour)
        img_crop = img_orig[y:y + h, x:x + w]
        images.append(img_crop)
    if not images:
        images.append(img_orig)
    return images


def remove_noise(img, thresh):
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(opening, kernel, iterations=3)
    return dilate


def resize_img(img):
    width = 64
    height = 64
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized


def func_sort(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    min_elem = np.min(box, axis=0)
    return min_elem[1]


# https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
def watershed(img):
    img_gray = img_to_grayscale(img)
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_inv = invert_image(thresh)
    kernel = np.ones((5, 5), np.uint8)
    sure_bg = cv2.dilate(img_inv, kernel, iterations=5)
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.73 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    markers1 = markers.astype(np.uint8)
    ret, img2 = cv2.threshold(markers1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    plt.show()
    _, contours, hierarchy = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return img, contours


def find_good_contours(contours, img):
    good_contours = []
    img_width = img.shape[0]
    img_height = img.shape[1]
    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        width, height = size
        img_area = img_width * img_height
        contour_area = width * height
        if contour_area > img_area / 13 and (width < 400 and height < 290):
            good_contours.append(contour)

    return good_contours


def remove_redundant_contours(good_contours):
    new_contours = []
    for contour in good_contours:
        if not is_redundant(new_contours, contour):
            new_contours.append(contour)

    return new_contours


def is_redundant(contours, contour):
    rect = cv2.minAreaRect(contour)
    box_points = cv2.boxPoints(rect)
    for contourIt in contours:
        rect2 = cv2.minAreaRect(contourIt)
        box_points2 = cv2.boxPoints(rect2)
        if box_points2[0][0] == box_points[0][0] and box_points2[0][1] == box_points[0][1] and \
                box_points2[1][0] == box_points[1][0] and box_points2[1][1] == box_points[1][1] and \
                box_points2[2][0] == box_points[2][0] and box_points2[2][1] == box_points[2][1] and \
                box_points2[3][0] == box_points[3][0] and box_points2[3][1] == box_points[3][1]:
            return True
    return False


def crop_by_contours(img_orig, contours):
    images = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        destination = np.array([[0, int(rect[1][1]) - 1],
                        [0, 0],
                        [int(rect[1][0]) - 1, 0],
                        [int(rect[1][0]) - 1, int(rect[1][1]) - 1]], 'float32')
        perspective_transform = cv2.getPerspectiveTransform(box, destination)
        warp_perspective = cv2.warpPerspective(img_orig, perspective_transform, (int(rect[1][0]), int(rect[1][1])))
        images.append(warp_perspective)
    return images


def extract_images(img):
    img, contours = watershed(img)
    good_contours = find_good_contours(contours, img)
    good_contours = remove_redundant_contours(good_contours)
    good_contours.sort(key=func_sort)
    return crop_by_contours(img, good_contours)
