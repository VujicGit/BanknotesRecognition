from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import Data.Data as data
import cv2
import matplotlib.pyplot as plt
import image_preprocesing as pp
import csv
import time

train_images_path = 'Dataset/train'
test_images_path_multiple = 'Dataset/test/multiple/'
test_images_path_single = 'Dataset/test/one'
saved_path = 'Saved'

train_data = data.get_images(train_images_path)
test_data = data.get_images(test_images_path_single)
test_data.extend(data.get_multiple(test_images_path_multiple))
sift = cv2.xfeatures2d.SIFT_create()
result_log = []


def calculate_train_desc():
    training = []
    print("***Generisanje treining deskriptora***")
    i = 0
    for data in train_data:
        img_orig = data[0]
        img_gs = pp.img_to_grayscale(img_orig)
        img_bin, thresh = pp.img_to_bin(img_gs)
        img_inv = pp.invert_image(img_bin)
        images = pp.crop(img_inv, img_orig)
        image_for_sift = images[0]
        # image_for_sift = pp.resize_img(image_for_sift)
        img_gs = pp.img_to_grayscale(image_for_sift)
        kp, desc = sift.detectAndCompute(img_gs, None)
        training.append((data, kp, desc))
        i = i + 1

    return training


def calculate_test_desc():
    test = []
    print("***Generisanje test deskriptora***")
    index = 0
    for data in test_data:
        img_orig = data[0]
        img_gs = pp.img_to_grayscale(img_orig)
        img_bin, thresh = pp.img_to_bin(img_gs)
        img_bin = pp.remove_noise(img_bin, thresh)
        img_inv = pp.invert_image(img_bin)
        images = pp.crop(img_inv, img_orig)
        for i in images:
            image_for_sift = i
            # image_for_sift = pp.resize_img(image_for_sift)
            img_gs = pp.img_to_grayscale(image_for_sift)
            kp, desc = sift.detectAndCompute(img_gs, None)
            test.append((data, kp, desc))
        index = index + 1
    return test


train_desc = calculate_train_desc()
test_desc = calculate_test_desc()

x_train = []
y_train = []
x_test = []
y_test = []

for i in train_desc:
    x_train.append(i[2])
    y_train.append(i[0][1])

for i in test_desc:
    x_test.append(i[2])
    y_test.append(i[0][1])


## https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
def get_matching(train_k, train_d, test_k, test_d):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(train_d, test_d, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    keypoints_len = min(len(test_k), len(train_k))

    if keypoints_len != 0:
        best_score = len(good) / keypoints_len
    else:
        best_score = 0
    return best_score


def predict():
    prediction_labels = []
    for test_desc_it in test_desc:
        best_match = 0
        for train_desc_it in train_desc:
            train_kp = train_desc_it[1]
            train_dsc = train_desc_it[2]
            test_kp = test_desc_it[1]
            test_dsc = test_desc_it[2]
            match = get_matching(train_kp, train_dsc, test_kp, test_dsc)
            if match > best_match:
                best_match = match
                predicted_label = train_desc_it[0][1]
        result_log.append((test_desc_it[0][2], test_desc_it[0][1], predicted_label))
        prediction_labels.append(predicted_label)
    return prediction_labels


def get_accuracy():
    print("***RACUNANJE TACNOSTI***")
    time_start = round(time.time() * 1000)
    results = predict()
    time_end = round(time.time() * 1000)
    print("Validation accuracy: ", round(accuracy_score(y_test, results) * 100, 2))
    print("Vreme izvrsavanja: " + str(time_end - time_start) + "ms")


def save_log():
    with open('log.csv', mode='w') as log_file:
        log_writer = csv.writer(log_file, delimiter=',')
        log_file.truncate()
        for i in result_log:
            log_writer.writerow([i[0], i[1], i[2]])
        log_file.close()


get_accuracy()
save_log()
