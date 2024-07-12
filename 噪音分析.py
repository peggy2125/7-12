import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.restoration import estimate_sigma
import os
import math

def analyze_noise(image_path,sumnoiselevel):
    print(f"Analyzing image: {image_path}")
    
    # 讀取圖像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image.")
        return
    print(f"Image loaded successfully. Shape: {img.shape}")
    
    # 1. 直方圖分析
    #hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    #print(f"Histogram calculated. Min: {hist.min()}, Max: {hist.max()}")
    
    # 2. 頻域分析
    #f = np.fft.fft2(img)
    #fshift = np.fft.fftshift(f)
    #magnitude_spectrum = 20*np.log(np.abs(fshift))
    #print(f"Frequency domain analysis completed. Spectrum range: {magnitude_spectrum.min()} to {magnitude_spectrum.max()}")
    
    # 3. 局部方差分析
    #local_var = ndimage.generic_filter(img, np.var, size=5)
    #print(f"Local variance calculated. Min: {local_var.min()}, Max: {local_var.max()}")
    
    # 4. 邊緣檢測
    #edges = cv2.Canny(img, 100, 200)
    #print(f"Edge detection completed. Number of edge pixels: {np.sum(edges > 0)}")
    
    # 5. 噪音估計
    try:
        sigma_est = estimate_sigma(img, channel_axis=None)
        print(f"Estimated Noise Level: {sigma_est:.2f}")
    except Exception as e:
        print(f"Error in noise estimation: {e}")
    sum = sumnoiselevel + sigma_est
    # 繪圖
    #plt.figure(figsize=(20, 10))
    #plt.subplot(231), plt.imshow(img, cmap='gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(232), plt.plot(hist)
    #plt.title('Histogram'), plt.xlim([0, 256])
    #plt.subplot(233), plt.imshow(magnitude_spectrum, cmap='gray')
    #plt.title('Frequency Domain'), plt.xticks([]), plt.yticks([])
    #plt.subplot(234), plt.imshow(local_var, cmap='jet')
    #plt.title('Local Variance'), plt.xticks([]), plt.yticks([])
    #plt.subplot(235), plt.imshow(edges, cmap='gray')
    #plt.title('Edge Detection'), plt.xticks([]), plt.yticks([])
    #plt.subplot(236), plt.text(0.5, 0.5, f'Estimated Noise Level: {sigma_est:.2f}', 
     #                          ha='center', va='center', fontsize=12)
    #plt.title('Noise Estimation'), plt.xticks([]), plt.yticks([])
    
    #plt.tight_layout()
    #plt.show()
    print("Analysis completed.")
    return sum
# 使用函數
#directory = 'Test_images/Slight under focus'
    # # Get a list of all tiff files

#files = [f for f in os.listdir(directory) if f.endswith('.tiff') and f != 'background.tiff']
sumnoiselevel = 0
#for image in files:
#    image_path = os.path.join(directory, image)
#    print(image_path)
#    sumnoiselevel = analyze_noise(image_path ,sumnoiselevel)
sumnoiselevel =analyze_noise('Test_images/Slight under focus/background.tiff' ,sumnoiselevel)
print("average noise level:", sumnoiselevel)#/192)


