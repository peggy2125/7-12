import cv2
import numpy as np
import math
import cv2
import numpy as np
from skimage.restoration import estimate_sigma

def add_noise(image_path, noise_type="gaussian", target_noise_level=1.468162461981323):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 估计原始图像的噪音水平
    original_noise_level = estimate_sigma(image, channel_axis=None)

    if noise_type == "gaussian":
        # 计算需要添加的噪音强度
        additional_noise_level = np.sqrt(target_noise_level**2 - original_noise_level**2)
        
        # 生成高斯噪音
        gaussian = np.random.normal(0, additional_noise_level, image.shape)
        noisy_image = np.clip(image + gaussian, 0, 255).astype(np.uint8)
    else:
        return image

    # 验证最终的噪音水平
    final_noise_level = estimate_sigma(noisy_image, channel_axis=None)
    print(f"Original noise level: {original_noise_level:.2f}")
    print(f"Target noise level: {target_noise_level:.2f}")
    print(f"Achieved noise level: {final_noise_level:.2f}")

    return noisy_image


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

def process_image(noisy_circle, blurred_bg):   #,output_dir):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) 
    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(noisy_circle, (5, 5), 0)
    # Background subtraction
    bg_sub = cv2.subtract(blurred_bg, blurred)
    # Apply threshold
    _, binary = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)
    return binary

background = cv2.imread('Test_images/background.tiff', cv2.IMREAD_GRAYSCALE)
blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
noisy_circle = add_noise('Test_images/perfect_circle.tiff', "gaussian")
cv2.imwrite('Test_images/noisy_perfect_circle.tiff', noisy_circle)
noisy_circle_1 = cv2.imread('Test_images/noisy_perfect_circle.tiff', cv2.IMREAD_GRAYSCALE)
image_1=process_image(noisy_circle_1,blurred_bg)
results = calculate_circle_metrics(image_1)
resultcon=calculate_contour_metrics(image_1)

if results:
    print("\n實心噪音圓形指標計算結果:")
    print(f"原始面積_噪音實心圓: {results['area_original']}")
    print(f"原始周長_噪音實心圓: {results['perimeter_original']:.2f}")
    print(f"原始圓度_噪音實心圓: {results['circularity_original']:.6f}")
    print(f"凸包面積_噪音實心圓: {results['area_hull']}")
    print(f"凸包周長_噪音實心圓: {results['perimeter_hull']:.2f}")
    print(f"凸包圓度_噪音實心圓: {results['circularity_hull']:.6f}")
    print(f"面積比_噪音實心圓: {results['area_ratio']:.6f}")
    print(f"圓度比_噪音實心圓: {results['circularity_ratio']:.6f}")
else:
    print("未能計算圓形指標。")
    
if resultcon:
    print("\n輪廓噪音圓形指標計算結果:")
    print(f"原始面積_噪音輪廓化圓: {resultcon['area_original_con']}")
    print(f"原始周長_噪音輪廓化圓: {resultcon['perimeter_original_con']:.2f}")
    print(f"原始圓度_噪音輪廓化圓: {resultcon['circularity_original_con']:.6f}")
    print(f"凸包面積_噪音輪廓化圓: {resultcon['area_hull_con']}")
    print(f"凸包周長_噪音輪廓化圓: {resultcon['perimeter_hull_con']:.2f}")
    print(f"凸包圓度_噪音輪廓化圓: {resultcon['circularity_hull_con']:.6f}")
    print(f"面積比_噪音輪廓化圓: {resultcon['area_ratio_con']:.6f}")
    print(f"圓度比_噪音輪廓化圓: {resultcon['circularity_ratio_con']:.6f}")
else:
    print("未能計算圓形指標。")