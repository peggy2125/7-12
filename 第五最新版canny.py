import cv2
import numpy as np
import time
import os
import math

def calculate_contour_metrics(contours):
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
    results = {
        "area_original": area_original,
        "area_hull": area_hull,
        "area_ratio": area_ratio,
        "circularity_original": circularity_original,
        "circularity_hull": circularity_hull,
        "circularity_ratio": circularity_ratio,
        "contour": cnt,
        "hull": hull
    }
    return results

def process_image(image_path, blurred_bg):   #,output_dir):
    # Load the image in grayscale
    start_time = time.perf_counter()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
    #cv2.imshow('raw', image)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) 
    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Background subtraction
    bg_sub = cv2.subtract(blurred_bg, blurred)
    # Apply threshold
    _, binary = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)
    # Erode and dilate to remove noise
    dilate1 = cv2.dilate(binary, kernel, iterations = 2)  # cv2.dilate() 函數對二值化後的圖像 binary 進行膨脹操作，kernel 定義膨脹的模式，iterations = 2 指定了膨脹操作重複進行的次數，總體可填補一些小的洞洞,使前景物體更加連續
    erode1 = cv2.erode(dilate1, kernel, iterations = 2)  #使用 cv2.erode() 函數對膨脹後的圖像 dilate1 進行腐蝕操作，去除一些小的雜訊
    erode2 = cv2.erode(erode1, kernel, iterations = 1)  #再次對 erode1 進行腐蝕操作,以進一步去除雜訊
    edges = cv2.Canny(erode2, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    results = calculate_contour_metrics(contours) #findcontour
    end_time = time.perf_counter()
    dif_time = end_time - start_time
    print(dif_time)
    return contours,results,image

def main(): 
    background = cv2.imread('Test_images/Slight under focus/background.tiff', cv2.IMREAD_GRAYSCALE)
    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
    directory = 'Test_images/Slight under focus'
    # # Get a list of all tiff files
    files = [f for f in os.listdir(directory) if f.endswith('.tiff') and f != 'background.tiff']
    for image in files:
        image_path = os.path.join(directory, image)
        print(image_path)
        contours, results, image = process_image(image_path, blurred_bg) #, output_dir)
        if results:
            print(f"Original area: {results['area_original']:.2f}")
            print(f"Convex Hull area: {results['area_hull']:.2f}")
            print(f"Area ratio (hull/original): {results['area_ratio']:.6f}")
            print(f"Original circularity: {results['circularity_original']:.6f}")
            print(f"Convex Hull circularity: {results['circularity_hull']:.6f}")
            print(f"Circularity ratio (hull/original): {results['circularity_ratio']:.6f}")
        # 繪製原始輪廓
            contour_image = np.zeros_like(image)
            cv2.drawContours(contour_image, contours, -1, 255, 1)
        cv2.imshow('Processed Image', contour_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()