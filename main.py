import os
import warnings

import PIL
import pandas as pd
from keras.src.utils import to_categorical

warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder


def extract_features(images):
    AllFeatures = []
    for image in (images):
        if ".DS_Store" not in image:
            img = keras.preprocessing.image.load_img(image, grayscale=True, target_size=(48, 48))
            img = np.array(img)
            AllFeatures.append(img)
    AllFeatures = np.array(AllFeatures)
    AllFeatures = AllFeatures.reshape(len(AllFeatures), 48, 48, 1)
    return AllFeatures


def load_dataset(directory):
    image_paths = []
    labels = []

    for label in os.listdir(directory):
        if not label.startswith('.'):
            label_dir = os.path.join(directory, label)
            for filename in os.listdir(label_dir):
                if not filename.startswith('.'):
                    image_path = os.path.join(label_dir, filename)
                    image_paths.append(image_path)
                    labels.append(label)


    return image_paths, labels


TRAIN_DIR = 'rizz/'
TEST_DIR = 'rizz/'

if __name__ == '__main__':
    train = pd.DataFrame()
    train['image'], train['label'] = load_dataset(TRAIN_DIR)
    train = train.sample(frac=1).reset_index(drop=True)
    train.head()

    test = pd.DataFrame()
    test['image'], test['label'] = load_dataset(TEST_DIR)
    test = test.sample(frac=1).reset_index(drop=True)
    test.head()

    LabelEncode = LabelEncoder()
    LabelEncode.fit(train['label'])
    y_train = LabelEncode.transform(train['label'])
    y_test = LabelEncode.transform(test['label'])

    num_classes = len(LabelEncode.classes_)

    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    model = keras.models.load_model('rizz model.h5')

    input1 = input("Input your face in absolute path form to determine rizz\n-1 to exit\n")
    while (input1 != "-1"): # While Loop for CLI
        imageUrl = input1

        # Make images grayscale
        img = PIL.Image.open(imageUrl).convert('L')

        # 48x48 size images
        img = img.resize((48, 48))

        # Change image to numpy array and expand dimensions
        ImageArray = np.array(img)

        ImageArray = np.expand_dims(ImageArray, axis=-1)
        ImageArray = np.expand_dims(ImageArray, axis=0)

        # Normalize the image
        ImageArray = ImageArray / 255.0

        # Finish all predictions
        predictions = model.predict(ImageArray)
        PredictionLabel = LabelEncode.inverse_transform([predictions.argmax()])[0]
        score = tf.nn.softmax(predictions[0])

        # Print out output
        print("Predicted Output:", PredictionLabel)
        print("Confidence:", 100 * np.max(score))
        input1 = input("Input your face in absolute path form to determine rizz\n-1 to exit\n")
