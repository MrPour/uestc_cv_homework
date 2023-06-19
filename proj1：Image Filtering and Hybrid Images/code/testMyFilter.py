
import os
import numpy as np
from skimage.transform import rescale
import matplotlib.pyplot as plt
import student
if __name__ == "__main__":
    test_image = student.load_image("../data/dog.bmp")
    test_image = rescale(test_image, 0.7, mode="reflect", multichannel=True)
    sobel_filter = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_image = student.my_imfilter(test_image, sobel_filter)
    plt.imshow(sobel_image)
    plt.show()
