import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def process_image(image_path, background_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
    cv2.imshow('raw', image)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imshow('blurred', blurred)

    # Background subtraction
    print(blurred.shape, blurred_bg.shape)
    bg_sub = cv2.subtract(blurred_bg, blurred)
    cv2.imshow('bg_sub', bg_sub)



# 计算直方图
    hist = cv2.calcHist([bg_sub], [0], None, [256], [0, 256])

# 创建图表
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(hist)
    plt.xlim([0, 256])

# 显示图表
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Set the directory containing your files
directory = 'Test_images/Slight under focus'
# Get a list of all tiff files
files = [f for f in os.listdir(directory) if f.endswith('.tiff')]
for image in files:
    image_path = os.path.join(directory, image)
    print(image_path)
    process_image(image_path, 'Test_images/Slight under focus/background.tiff')