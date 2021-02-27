from numpy import string_
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.python.keras import Sequential
import cv2

import Data.Data as data
import random
import image_preprocesing as pp
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

labels = ['5e', '10e', '20e', '50e', '100e', '200e', '500e', '10d', '20d', '50d', '100d', '200d', '500d', '1000d',
          '2000d']

train_images_path = 'Dataset/train'
test_images_path_multiple = 'Dataset/test/multiple/'
test_images_path_single = 'Dataset/test/single'
train_data = data.get_images(train_images_path)
test_data = data.get_images(test_images_path_single)
test_data.extend(data.get_multiple(test_images_path_multiple))


def get_train_tuples():
    train_tuples = []
    for i in train_data:
        img = i[0]
        label = i[1]
        resized_img = pp.resize_img(img)
        train_tuples.append((resized_img, label))
    return train_tuples


def get_test_data():
    x_test = []
    y_test = []
    for i in test_data:
        img = i[0]
        label = i[1]
        img_resized = pp.resize_img(img)
        x_test.append(img_resized)
        y_test.append(label)
    return x_test, y_test


def get_train_data(train_data):
    x_train = []
    y_train = []
    random.shuffle(train_data)
    for i in train_data:
        img = i[0]
        label = i[1]
        x_train.append(img)
        y_train.append(label)

    return x_train, y_train


mapping = {}


def initialize_encoder():
    for i in range(len(labels)):
        mapping[labels[i]] = i
    return mapping


def encode_label(labels):
    for i in range(len(labels)):
        labels[i] = mapping[labels[i]]
    return to_categorical(labels)


def decode_label(vector):
    index = np.argmax(vector)
    return labels[index]


mapped = initialize_encoder()


def save_model(model):
    model_json = model.to_json()
    with open("serialization/trained_model2.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("serialization/trained_model2.h5")


def data_augmentation_step2(train_tupples):
    print("***Usao u step2***")
    counter = 0
    augmented = []
    for i in train_tupples:
        img = i[0]
        label = i[1]
        new_img = random_flip(img)
        new_img = random_translation(new_img)
        new_img = random_rotation(new_img)
        new_img = random_zoom_in(new_img)
        new_img = random_zoom_out(new_img)
        augmented.append((new_img, label))
        print("Zavrsio: " + str(counter))
        counter = counter + 1
    return augmented


def data_augmentation_step1(train_tuples):
    print("***Usao u step 1***")
    flip = []
    zoom_in = []
    zoom_out = []
    rotation = []
    translation = []
    for i in train_tuples:
        new_img = random_flip(i[0])
        label = i[1]
        flip.append((new_img, label))
    for i in train_tuples:
        new_img = random_zoom_in(i[0])
        label = i[1]
        zoom_in.append((new_img, label))
    for i in train_tuples:
        new_img = random_zoom_out(i[0])
        label = i[1]
        zoom_out.append((new_img, label))
    for i in train_tuples:
        new_img = random_rotation(i[0])
        label = i[1]
        rotation.append((new_img, label))
    for i in train_tuples:
        new_img = random_translation(i[0])
        label = i[1]
        translation.append((new_img, label))

    train_tuples.extend(flip)
    train_tuples.extend(zoom_in)
    train_tuples.extend(zoom_out)
    train_tuples.extend(rotation)
    train_tuples.extend(translation)
    return train_tuples


def random_flip(img):
    img = tf.expand_dims(img, 0)
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")
    ])
    new_img = model(img)
    return new_img[0]


def random_rotation(img):
    img = tf.expand_dims(img, 0)
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomRotation(factor=1)
    ])
    new_img = model(img)
    return new_img[0]


def random_zoom_in(img):
    img = tf.expand_dims(img, 0)
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomZoom(-0.3)
    ])
    new_img = model(img)
    return new_img[0]


def random_zoom_out(img):
    img = tf.expand_dims(img, 0)
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomZoom(0.3)
    ])
    new_img = model(img)
    return new_img[0]


def random_translation(img):
    img = tf.expand_dims(img, 0)
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.2, 0.3), width_factor=(-0.2, 0.3))
    ])
    new_img = model(img)
    return new_img[0]


# Arhitektura modela preuzeta iz knjige Python Machine Learning
def create_model():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(15, activation='softmax'))
    model.build(input_shape=(None, 64, 64, 3))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model


def train_model():
    print("***Treniranje modela***")
    model = create_model()
    augmented_data = data.get_data('Saved/train.npy')
    x_train, y_train = get_train_data(augmented_data)
    y_train_encoded = encode_label(y_train)
    x_train = np.array(x_train, 'float32')
    y_train = np.array(y_train_encoded, 'float32')
    model.fit(x=x_train, y=y_train, epochs=1000, verbose=1, steps_per_epoch=len(x_train) / 32)
    print("***Treniranje modela zavrseno***")
    test_loss, test_accuracy = model.evaluate(x_train, y_train)
    print("Test accuracy: " + str(test_accuracy))
    save_model(model)


def load_trained_model():
    try:
        json_file = open('serialization/trained_model2.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        cnn = tf.keras.models.model_from_json(loaded_model_json)
        cnn.load_weights('serialization/trained_model2.h5')
        return cnn
    except Exception as e:
        return None


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def predict():
    model = load_trained_model()
    x_test, y_test = get_test_data()
    x_test = np.array(x_test, 'float32')
    y_predict = model.predict(x_test)
    winners = []
    for i in y_predict:
        winners.append(labels[winner(i)])

    accuracy = accuracy_score(winners, y_test)
    print("Accuracy score: " + str(round(accuracy * 100, 2)) + "%")


predict()
