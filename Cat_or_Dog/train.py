import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from pathlib import Path
import argparse

SAVE_DIR = "backup"  # Save directory for backup weights during the training


class DogCatClassifier:
    """
    Image classifier for dog and cat pictures using Deep Learning
    Convolutionnal Neural Network
    """

    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    BATCH_SIZE = 64

    def __init__(self, data_dir="data", epochs=1):
        """
        :param data_dir: directory of the data
        :param epochs: number of epochs for the training
        """
        self.epochs = epochs
        self.data_dir = data_dir

        # Load data and labels
        self.X = sorted(os.listdir(self.data_dir))  # Files names of the images
        self.y = np.empty(len(self.X), dtype=str)  # Labels
        self.y[np.char.startswith(self.X, "cat")] = "c"
        self.y[np.char.startswith(self.X, "dog")] = "d"

        self.model = DogCatClassifier._load_model()

    def fit(self, folder):
        """Fit the model using the data in the selected directory"""
        train_set, val_set, test_set = self._gen_data()

        # callback object to save weights during the training
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(SAVE_DIR, "weights-{epoch:03d}.ckpt"),
            save_weights_only=True,
            verbose=1,
        )

        # Fit the model
        history = self.model.fit(
            train_set,
            epochs=self.epochs,
            validation_data=val_set,
            callbacks=[cp_callback],
        )

        # Show the predictions on the testing set
        result = self.model.evaluate(test_set, batch_size=self.BATCH_SIZE)
        print(
            "Testing set evaluation:",
            dict(zip(self.model.metrics_names, result)),
        )

        # Save model information
        self.model.save(folder)

        # Plot training results
        epochs_range = range(self.epochs)

        # Accuracy in training and validation sets as the training goes
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="Training Accuracy")
        plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        # Loss in training and validation sets as the training goes
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")

        plt.savefig(os.path.join(SAVE_DIR, "results.png"))

    @classmethod
    def _load_model(cls):
        """Build a CNN model for image classification"""
        model = Sequential()

        # 2D Convolutional layer
        model.add(
            Conv2D(
                128,  # Number of filters
                (3, 3),  # Padding size
                input_shape=(
                    cls.IMG_HEIGHT,
                    cls.IMG_WIDTH,
                    3,
                ),  # Shape of the input images
                activation="relu",  # Output function of the neurons
                padding="same",
            )
        )  # Behaviour of the padding region near the borders
        # 2D Pooling layer to reduce image shape
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Transform 2D input shape into 1D shape
        model.add(Flatten())
        # Dense layer of fully connected neurons
        model.add(Dense(128, activation="relu"))
        # Dropout layer to reduce overfitting, the argument is the proportion of random neurons ignored in the training
        model.add(Dropout(0.2))
        # Output layer
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy",  # Loss function for binary classification
            optimizer=RMSprop(
                lr=1e-3
            ),  # Optimizer function to update weights during the training
            metrics=["accuracy", "AUC"],
        )  # Metrics to monitor during training and testing

        # Print model summary
        model.summary()

        return model

    def _gen_data(self):
        """Split the data set into training, validation and testing sets"""

        # Split data into training+validation and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        df_train = pd.DataFrame({"filename": X_train, "class": y_train})
        df_test = pd.DataFrame({"filename": X_test, "class": y_test})

        # use data generators as input for the model
        train_datagen = ImageDataGenerator(
            rescale=1 / 255,  # Divide input values by 255 so it ranges between 0 and 1
            # The images are converted from RGB to BGR, then each color channel is
            # zero-centered with respect to the ImageNet dataset, without scaling.
            preprocessing_function=preprocess_input,
            validation_split=0.2,  # Size of the validation set
            horizontal_flip=True,  # Includes random horizontal flips in the data set
            shear_range=0.2,  # Includes random shears in the data set
            height_shift_range=0.2,  # Includes random vertical shifts in the data set
            width_shift_range=0.2,  # Includes random horizontal shifts in the data set
            zoom_range=0.2,  # Includes random zooms in the data set
            rotation_range=30,  # Includes random rotations in the data set
            # Filling methods for undefined regions upon data augmentation
            fill_mode="nearest",
        )
        test_datagen = ImageDataGenerator(
            rescale=1 / 255, preprocessing_function=preprocess_input
        )

        # Load images in the data generators
        train_data_generator = train_datagen.flow_from_dataframe(
            df_train,
            # Directory in which the files can be found
            directory=self.data_dir,
            # Column name for the image names
            x_col="filename",
            # Column name for the labels
            y_col="class",
            # Type of subset
            subset="training",
            # Shuffle the data to avoid fitting the image order
            shuffle=True,
            # batch size
            batch_size=self.BATCH_SIZE,
            # Classification mode
            class_mode="binary",
            # Target size of the images
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
        )
        valid_data_generator = train_datagen.flow_from_dataframe(
            df_train,
            directory=self.data_dir,
            x_col="filename",
            y_col="class",
            subset="validation",
            shuffle=True,
            batch_size=self.BATCH_SIZE,
            class_mode="binary",
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
        )
        test_data_generator = test_datagen.flow_from_dataframe(
            df_test,
            directory=self.data_dir,
            x_col="filename",
            y_col="class",
            shuffle=False,
            batch_size=self.BATCH_SIZE,
            class_mode="binary",
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
        )

        return train_data_generator, valid_data_generator, test_data_generator


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CNN Trainer for the Cat or Dog app.")

    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        help="Destination folder to save the model after training ends.",
        default="Custom",
    )
    args = parser.parse_args()

    if Path(f"model_{args.folder}").is_dir():
        print(f"Folder model_{args.folder} already exists do you want to overwrite ?")
        y = input('Type "Yes" or "No": ')
        if y != "Yes":
            print("Aborting.")
            sys.exit()

    clf = DogCatClassifier()
    clf.fit(Path(f"model_{args.folder}"))
