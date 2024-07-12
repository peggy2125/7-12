import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import time

def load_images(directory):
    images = {}
    for filename in os.listdir(directory):
        if filename.endswith('.tiff') and filename != 'background.tiff':
            path = os.path.join(directory, filename)
            images[filename] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return images

def apply_gaussian_blur(images, kernel_size=(5, 5)):
    with ThreadPoolExecutor() as executor:
        return {name: executor.submit(cv2.GaussianBlur, img, kernel_size, 0).result() 
                for name, img in images.items()}

def subtract_background(images, background):
    with ThreadPoolExecutor() as executor:
        return {name: executor.submit(cv2.subtract, background, img).result() 
                for name, img in images.items()}

def apply_threshold(images, thresh=10):
    with ThreadPoolExecutor() as executor:
        return {name: executor.submit(lambda x: cv2.threshold(x, thresh, 255, cv2.THRESH_BINARY)[1], img).result() 
                for name, img in images.items()}

def apply_morphology(images, operation, kernel, iterations):
    with ThreadPoolExecutor() as executor:
        return {name: executor.submit(operation, img, kernel, iterations=iterations).result() 
                for name, img in images.items()}

def apply_canny(images, threshold1=50, threshold2=150):
    with ThreadPoolExecutor() as executor:
        return {name: executor.submit(cv2.Canny, img, threshold1, threshold2).result() 
                for name, img in images.items()}

def trace_contours(images):
    with ThreadPoolExecutor() as executor:
        return {name: executor.submit(lambda x: cv2.findContours(x, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0], img).result() 
                for name, img in images.items()}

def draw_contours(images, contours):
    with ThreadPoolExecutor() as executor:
        return {name: executor.submit(lambda img, cnts: cv2.drawContours(np.zeros_like(img), cnts, -1, (255), 1), images[name], cnts).result() 
                for name, cnts in contours.items()}

def process_images(directory):
    # Load images and background
    images = load_images(directory)
    background = cv2.imread(os.path.join(directory, 'background.tiff'), cv2.IMREAD_GRAYSCALE)
    background = cv2.GaussianBlur(background, (5, 5), 0)

    # Create kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    start_time = time.perf_counter()

    # Apply processing steps
    blurred = apply_gaussian_blur(images)
    bg_sub = subtract_background(blurred, background)
    binary = apply_threshold(bg_sub)
    dilate1 = apply_morphology(binary, cv2.dilate, kernel, 2)
    erode1 = apply_morphology(dilate1, cv2.erode, kernel, 2)
    erode2 = apply_morphology(erode1, cv2.erode, kernel, 1)
    dilate2 = apply_morphology(erode2, cv2.dilate, kernel, 1)
    edges = apply_canny(dilate2)
    contours = trace_contours(edges)
    contour_images = draw_contours(images, contours)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.6f} seconds")
    print(f"Average time per image: {total_time/len(images):.6f} seconds")
    return contour_images

# Main execution
directory = 'Test_images/Slight under focus'
files = [f for f in os.listdir(directory) if f.endswith('.tiff') and f != 'background.tiff']

start_time = time.perf_counter()
processed_images = process_images(directory)
end_time = time.perf_counter()

total_time = end_time - start_time
print(f"Total execution time (including loading and processing): {total_time:.6f} seconds")
print(f"average time:{total_time/len(files):.6f}sec")
# Display results
for name, img in processed_images.items():
    cv2.imshow(name, img)
    print(f"Showing image: {name}. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()