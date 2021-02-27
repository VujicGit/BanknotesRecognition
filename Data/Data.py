import os
import cv2
import numpy as np
import csv
import image_preprocesing as pp


saved_data = 'Saved'
train_images_path = '../Dataset/train'
test_images_path_one = '../Dataset/test/one'
test_images_path_multiple = '../Dataset/test/multiple'


def get_class(path):
    tokens = str(path).split("\\")
    return tokens[1]


def get_images(path):
    training_images = []
    for classes in os.listdir(path):
        joined = os.path.join(path, classes)
        for imgIt in os.listdir(joined):
            joined2 = os.path.join(joined, imgIt)
            img = cv2.imread(joined2)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            class_name = get_class(joined)
            training_images.append((img, class_name, imgIt))
    return training_images


def get_multiple(path):
    test_tupples = []
    with open(path + 'labels.csv') as labels_file:
        csv_reader = csv.reader(labels_file, delimiter=',')
        for row in csv_reader:
            img_name = row[0]
            img_path = path + img_name
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cropped_images = pp.extract_images(img)
            for i in range(1, len(row)):
                class_name = row[i]
                test_tupples.append((cropped_images[i-1], class_name, img_name))
    return test_tupples


def save_train_data(path, train_data):
    np.save(path, np.array(train_data), allow_pickle=True)


def save_test_data(path, test_data):
    np.save(path, np.array(test_data), allow_pickle=True)


def get_data(path):
    saved = np.load(path, allow_pickle=True)
    return saved






