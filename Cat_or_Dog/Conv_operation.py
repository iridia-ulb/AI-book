import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy


class Transformations:
    """
    Implementation of a set of convolutions and pooling operations enabling to
    visualize the transformations applied to an image
    """

    def __init__(self, imgPath):
        self.padding = "SAME"
        self.stride = (2, 2)
        self.img = Image.open(imgPath)

    def choose_kernel(self, kernel_name="identity"):
        """Enables the user to choose the kernel.
        INPUT: str: name of the kernel, default='identity'
            kernels have been adapted from https://setosa.io/ev/image-kernels/"""
        if kernel_name == "sharpen":
            # Sharpen Kernel
            a = np.zeros([3, 3, 3, 3])
            a[1, 1, :, :] = 5
            a[0, 1, :, :] = -1
            a[1, 0, :, :] = -1
            a[2, 1, :, :] = -1
            a[1, 2, :, :] = -1
        elif kernel_name == "blur":
            # Blur Kernel
            a = np.full([3, 3, 3, 3], 0.0625)
            a[1, 1, :, :] = 0.25
            a[0, 1, :, :] = 0.125
            a[1, 0, :, :] = 0.125
            a[2, 1, :, :] = 0.125
            a[1, 2, :, :] = 0.125
        elif kernel_name == "bottom sobel":
            # Bottom sobel kernel
            a = np.zeros([3, 3, 3, 3])
            a[0, 2, :, :] = -1
            a[0, 1, :, :] = -2
            a[0, 0, :, :] = -1
            a[2, 0, :, :] = 1
            a[2, 1, :, :] = 2
            a[2, 2, :, :] = 1
        elif kernel_name == "emboss kernel":
            # Emboss kernel
            a = np.ones([3, 3, 3, 3])
            a[0, 2, :, :] = 0
            a[0, 1, :, :] = -1
            a[0, 0, :, :] = -2
            a[2, 0, :, :] = 0
            a[2, 2, :, :] = 2
            a[1, 0, :, :] = -1
        elif kernel_name == "identity":
            # Identity kernel
            a = np.zeros([3, 3, 3, 3])
            a[1, 1, :, :] = 1
        elif kernel_name == "left sobel":
            # Left sobel
            a = np.zeros([3, 3, 3, 3])
            a[0, 2, :, :] = -1
            a[0, 0, :, :] = 1
            a[1, 0, :, :] = 2
            a[2, 0, :, :] = 1
            a[1, 2, :, :] = -2
            a[2, 2, :, :] = -1
        elif kernel_name == "outline":
            # Outline kernel
            a = np.full([3, 3, 3, 3], -1)
            a[1, 1, :, :] = 8
        elif kernel_name == "right sobel":
            # Right sobel kernel
            a = np.zeros([3, 3, 3, 3])
            a[0, 2, :, :] = 1
            a[0, 0, :, :] = -1
            a[1, 0, :, :] = -2
            a[2, 0, :, :] = -1
            a[1, 2, :, :] = 2
            a[2, 2, :, :] = 1
        elif kernel_name == "top sobel":
            # Top sobel kernel
            a = np.zeros([3, 3, 3, 3])
            a[0, 2, :, :] = 1
            a[0, 1, :, :] = 2
            a[0, 0, :, :] = 1
            a[2, 0, :, :] = -1
            a[2, 1, :, :] = -2
            a[2, 2, :, :] = -1
        # Transforms the kernel to floats
        self.kernel = tf.constant(a, dtype=tf.float32)

    def multilayer(self, n_layers):
        """Main function for performing the transformations.
        Input: number of layers of convolution + pooling you want to perform
        OUTPUT: path to the final image"""

        for i in range(1, n_layers + 1):
            # For the first layer, take the input image
            if i == 1:
                # Convolution
                self.convolution(i, self.img)
                # Use the output of the convolution as input of the pooling layer
                img = Image.open(str("conv" + str(i) + ".png"))
                # Pooling
                fname = self.pool(img.convert("RGB"), i)
            else:
                # Use the output of the pooling as input of the convolution layer
                img = Image.open(str("conv_pool" + str(i - 1) + ".png"))
                # Convolution
                self.convolution(i, img.convert("RGB"))
                # Use the output of the convolution as input of the pooling layer
                img = Image.open(str("conv" + str(i) + ".png"))
                # Pooling
                fname = self.pool(img.convert("RGB"), i)
        return fname

    def img_reshape(self, img):
        """Reshape the image to have a leading one dimension
        INPUT: image (pillow object)
        OUTPUT: reshaped image (np array)"""

        # reshape image to have a leading 1 dimension
        img = numpy.asarray(img, dtype="float32") / 256.0
        img_shape = img.shape
        img_reshaped = img.reshape(1, img_shape[0], img_shape[1], 3)
        return img_reshaped

    def convolution(self, layer, img):
        """Convolution operation
        INPUT:  layer: int giving the number of convolutions+pooling operations done previously
                img: pillow object of the image to convolve
        OUTPUT: saves the convolved image to a file 'conv + layer + .png'
                return: str: the path to the convolved image"""

        # Reshape the pillow image to a numpy array
        img_reshaped = self.img_reshape(img)
        # Transform the kernel to a usable format to pass through the "conv2d" function
        w = tf.compat.v1.get_variable(
            "w", initializer=tf.compat.v1.to_float(self.kernel)
        )
        # Convolution operation
        conv = tf.nn.conv2d(
            input=img_reshaped,
            filters=w,
            strides=self.stride,
            padding=self.padding,
        )
        # Name of the file in which the image is stored
        fname = str("conv" + str(layer) + ".png")
        # Saving the transformed, grayscale image
        plt.imsave(
            fname,
            np.array(conv[0, :, :, 0], dtype="float32"),
            cmap=plt.get_cmap("gray"),
        )

        return fname

    def pool(self, img, layer):
        """Maxooling operation preceded by applying a sigmoid activation function
        INPUT:  img: pillow object of the image to pool
                layer: int giving the number of convolutions+pooling operations done previously
        OUTPUT: saves the pooled image to a file 'conv_pool + layer + .png'
                return: str: the path to the pooled image"""

        # Reshape the image
        img = self.img_reshape(img)
        # Apply the sigmoid activation function
        sig = tf.sigmoid(img)
        # Maxpooling
        max_pool = tf.nn.max_pool(
            sig, ksize=[1, 2, 2, 1], strides=self.stride, padding=self.padding
        )
        # Name of the file in which the image is stored
        fname = str("conv_pool" + str(layer) + ".png")
        # Save the pooled, grayscale image
        plt.imsave(
            fname,
            np.array(max_pool[0, :, :, 0], dtype="float32"),
            cmap=plt.get_cmap("gray"),
        )

        return fname
