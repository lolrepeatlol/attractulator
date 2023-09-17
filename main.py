import pathlib

import pandas as pd
import os
import seaborn
import warnings

from keras.src.utils import to_categorical
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

def extract_features(images):
    features = []
    for image in (images):
        print(image)
        if image[13] != '.' and image[10] != '.':
            img = keras.preprocessing.image.load_img(image, grayscale=True, target_size = (48,48))
            img = np.array(img)
            features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

def load_dataset(directory):
    image_paths = []
    labels = []

    for label in os.listdir(directory):
        if not label.startswith('.'):
            for filename in os.listdir(directory + label):
                image_path = os.path.join(directory, label, filename)
                image_paths.append(image_path)
                labels.append(label)

            print(label, "Completed")

    return image_paths, labels

TRAIN_DIR = 'rizz/'
TEST_DIR = 'rizz/'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train = pd.DataFrame()
    train['image'], train['label'] = load_dataset(TRAIN_DIR)
    train = train.sample(frac=1).reset_index(drop=True)
    train.head()

    test = pd.DataFrame()
    test['image'], test['label'] = load_dataset(TEST_DIR)
    test = test.sample(frac=1).reset_index(drop=True)
    test.head()

    train_features = extract_features(train['image'])
    test_features = extract_features(test['image'])

    x_train = train_features/255.0
    x_test = test_features/255.0

    le = LabelEncoder()
    le.fit(train['label'])
    y_train = le.transform(train['label'])
    y_test = le.transform(test['label'])

    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)

    input_shape = (48,48,1)
    output_class = 2

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", input_shape=(48, 48, 1)),
        layers.RandomFlip("vertical", input_shape=(48, 48, 1)),
        layers.RandomRotation(1),
        layers.RandomZoom(1),
    ])
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(output_class, activation='softmax', name="outputs")
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    epochs = 2
    print(x_train.shape)
    print(y_train.shape)
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size = 128,
        epochs=epochs,
        validation_data=(x_test,y_test)
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    imageUrl = "/Users/eren-mac/PycharmProjects/rizz calc /rizz/rizz/1.png"
    # sunflower_path = tf.keras.utils.get_file('abs', origin=imageUrl)

    img = tf.keras.utils.load_img(
        imageUrl, target_size=(48, 48)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array[0].reshape(1, 48, 48, 1))
    prediction_label = le.inverse_transform([predictions.argmax()])[0]

    score = tf.nn.softmax(predictions[0])
    print("Predicted Output:", prediction_label)

    if class_names[np.argmax(score)] != "not a person":
        print(
            "This image most likely has {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
    else:
        print(
            "This image most likely is {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
