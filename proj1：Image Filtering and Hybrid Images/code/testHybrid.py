import os
import numpy as np
from skimage.transform import rescale
import matplotlib.pyplot as plt
import student
if __name__ == "__main__":
    image1 = student.load_image("../data/dog.bmp")
    image2 = student.load_image("../data/cat.bmp")

    #展示原图
    plt.figure(figsize=(3, 3))
    plt.imshow((image1 * 255).astype(np.uint8))
    plt.show()
    plt.figure(figsize=(3, 3))
    plt.imshow((image2 * 255).astype(np.uint8))
    plt.show()

    #展示滤镜结果
    cutoff_frequency = 7
    low_frequencies, high_frequencies, hybrid_image = student.gen_hybrid_image(image1,image2,cutoff_frequency)
    plt.figure()
    plt.imshow((low_frequencies * 255).astype(np.uint8))
    plt.show()

    plt.figure()
    plt.imshow(((high_frequencies + 0.5) * 255).astype(np.uint8))
    plt.show()

    #展示混合图片
    vis = student.vis_hybrid_image(hybrid_image)
    plt.figure(figsize=(20, 20))
    plt.imshow(vis)
    plt.show()