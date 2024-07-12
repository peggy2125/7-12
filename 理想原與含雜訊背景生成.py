
import cv2
import numpy as np
import time
import os
import math
from scipy import ndimage
from skimage.restoration import estimate_sigma

def create_background(shape, mean=128, std=20, target_noise_level=1.48):
    """創建更真實的背景並調整噪音水平"""
    # 創建基礎噪聲
    background = np.random.normal(mean, std, shape).astype(np.float32)
    
    # 添加一些大尺度變化
    x, y = np.meshgrid(np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0]))
    gradient = 25 * (x + y)
    background += gradient
    
    # 確保值在0-255範圍內
    background = np.clip(background, 0, 255).astype(np.uint8)
    
    # 估計原始背景的噪音水平
    original_noise_level = estimate_sigma(background, channel_axis=None)
    
    # 計算需要添加的噪音強度
    additional_noise_level = np.sqrt(target_noise_level**2 - original_noise_level**2)
    
    # 生成高斯噪音
    gaussian = np.random.normal(0, additional_noise_level, background.shape)
    noisy_background = np.clip(background + gaussian, 0, 255).astype(np.uint8)
    
    # 驗證最終的噪音水平
    final_noise_level = estimate_sigma(noisy_background, channel_axis=None)
    print(f"Original noise level: {original_noise_level:.2f}")
    print(f"Target noise level: {target_noise_level:.2f}")
    print(f"Achieved noise level: {final_noise_level:.2f}")
    
    return background

def create_circle(area, margin):
    radius = math.sqrt(area / math.pi)    # 計算圓的半徑，使用面積公式：area = π * r^2
    diameter = int(2 * radius + 1)    # 計算直徑並取整，加1確保是奇數以便有明確的中心點
    image_size = diameter + 2 * margin     # 計算最終圖像的大小，包括圓形和額外的邊距
    center = (image_size // 2, image_size // 2)     # 計算圓心坐標，位於圖像的正中央
    image = np.zeros((image_size, image_size), dtype=np.uint8)     # 創建一個全黑的正方形圖像，大小為 image_size x image_size
    
    for y in range(image_size):
        for x in range(image_size):    
            # 檢查每個像素是否在圓內
            # 使用圓的方程式：(x - center_x)^2 + (y - center_y)^2 <= r^2
            if (x - center[0])**2 + (y - center[1])**2 <= radius**2:
                image[y, x] = 255 # 如果在圓內，將該像素設置為白色（255）
    
    return image


    
def calculate_circle_metrics(circle_image):
    # 計算面積
    area_original = np.sum(circle_image == 255)
    
    # 找到非零像素的坐標
    coords = np.column_stack(np.where(circle_image > 0))
    
    # 計算周長（使用最外圍的點）
    hull_original = cv2.convexHull(coords)
    perimeter_original = cv2.arcLength(hull_original, True)
    
    if area_original <= 1e-6 or perimeter_original <= 1e-6:
        print(f"Invalid measurements: area={area_original}, perimeter={perimeter_original}")
        return None
    
    # 計算圓度
    circularity_original = (4 * math.pi * area_original) / (perimeter_original ** 2)
    
    # 計算凸包
    hull = cv2.convexHull(coords)
    area_hull = cv2.contourArea(hull)
    perimeter_hull = cv2.arcLength(hull, True)
    
    if area_hull <= 1e-6 or perimeter_hull <= 1e-6:
        print(f"Invalid hull measurements: area={area_hull}, perimeter={perimeter_hull}")
        return None
    
    circularity_hull = (4 * math.pi * area_hull) / (perimeter_hull ** 2)
    
    # 計算比值
    area_ratio = area_hull / area_original
    circularity_ratio = circularity_hull / circularity_original

    results = {
        "area_original": area_original,
        "perimeter_original": perimeter_original,
        "circularity_original": circularity_original,
        "area_hull": area_hull,
        "perimeter_hull": perimeter_hull,
        "circularity_hull": circularity_hull,
        "area_ratio": area_ratio,
        "circularity_ratio": circularity_ratio
    }

    return results

def calculate_contour_metrics(circle_image):
    contours, _ = cv2.findContours(circle_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    
    # 計算原始輪廓的面積和圓度
    area_original = cv2.contourArea(cnt)
    perimeter_original = cv2.arcLength(cnt, True)
    if area_original <= 1e-6 or perimeter_original <= 1e-6:
        print(f"Invalid contour measurements: area={area_original}, perimeter={perimeter_original}")
        return None
    
    circularity_original = float(2 * math.sqrt((math.pi) * area_original)) / perimeter_original
    
    # 計算凸包
    hull = cv2.convexHull(cnt)
    
    # 計算凸包的面積和圓度
    area_hull = cv2.contourArea(hull)
    perimeter_hull = cv2.arcLength(hull, True)
    
    if area_hull <= 1e-6 or perimeter_hull <= 1e-6:
        print(f"Invalid hull measurements: area={area_hull}, perimeter={perimeter_hull}")
        return None
    
    circularity_hull = float(2 * math.sqrt((math.pi) * area_hull)) / perimeter_hull
    # 計算比值
    area_ratio = area_hull / area_original
    circularity_ratio = circularity_hull / circularity_original

    resultcon = {
        "area_original_con": area_original,
        "area_hull_con": area_hull,
        "area_ratio_con": area_ratio,
        "circularity_original_con": circularity_original,
        "circularity_hull_con": circularity_hull,
        "circularity_ratio_con": circularity_ratio,
        "perimeter_original_con": perimeter_original,
        "perimeter_hull_con": perimeter_hull
    }

    return resultcon

# 指定圓的面積（以像素為單位）
area = 100000  # 例如，100000像素的面積
margin = 30
# 創建圓
circle_image = create_circle(area, margin)
# 創建背景
background_image = create_background(circle_image.shape,target_noise_level=1.48)
# 指定保存路徑
circle_save_path = "C:/Users/USER/RT-DC-master/Test_images/perfect_circle.tiff"
background_save_path = "C:/Users/USER/RT-DC-master/Test_images/background.tiff"
actual_area = np.sum(circle_image == 255)
print(f"請求的面積: {area}")
print(f"實際的面積: {actual_area}")

# 保存圖像
cv2.imwrite(circle_save_path, circle_image)
cv2.imwrite(background_save_path, background_image)

print(f"圓形圖像已保存到: {circle_save_path}")
print(f"背景圖像已保存到: {background_save_path}")

results = calculate_circle_metrics(circle_image)
resultcon=calculate_contour_metrics(circle_image)

if results:
    print("\n實心圓形指標計算結果:")
    print(f"請求的面積: {area}")
    print(f"原始面積_實心圓: {results['area_original']}")
    print(f"原始周長_實心圓: {results['perimeter_original']:.2f}")
    print(f"原始圓度_實心圓: {results['circularity_original']:.6f}")
    print(f"凸包面積_實心圓: {results['area_hull']}")
    print(f"凸包周長_實心圓: {results['perimeter_hull']:.2f}")
    print(f"凸包圓度_實心圓: {results['circularity_hull']:.6f}")
    print(f"面積比_實心圓: {results['area_ratio']:.6f}")
    print(f"圓度比_實心圓: {results['circularity_ratio']:.6f}")
else:
    print("未能計算圓形指標。")
    
if resultcon:
    print("\n輪廓圓形指標計算結果:")
    print(f"原始面積_輪廓化圓: {resultcon['area_original_con']}")
    print(f"原始周長_輪廓化圓: {resultcon['perimeter_original_con']:.2f}")
    print(f"原始圓度_輪廓化圓: {resultcon['circularity_original_con']:.6f}")
    print(f"凸包面積_輪廓化圓: {resultcon['area_hull_con']}")
    print(f"凸包周長_輪廓化圓: {resultcon['perimeter_hull_con']:.2f}")
    print(f"凸包圓度_輪廓化圓: {resultcon['circularity_hull_con']:.6f}")
    print(f"面積比_輪廓化圓: {resultcon['area_ratio_con']:.6f}")
    print(f"圓度比_輪廓化圓: {resultcon['circularity_ratio_con']:.6f}")
else:
    print("未能計算圓形指標。")
