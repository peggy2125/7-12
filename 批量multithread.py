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
    #dilate2 = apply_morphology(erode2, cv2.dilate, kernel, 1)
    edges = apply_canny(erode2)

    end_time = time.perf_counter()
    print(f"Total processing time: {end_time - start_time:.6f} seconds")
    print(f"Average time per image: {(end_time - start_time)/len(images):.6f} seconds")
    return edges

# Main execution
directory = 'Test_images/Slight under focus'
files = [f for f in os.listdir(directory) if f.endswith('.tiff') and f != 'background.tiff']
processed_images = process_images(directory)

# Display results
for name, img in processed_images.items():
    cv2.imshow(name, img)
    print(f"Showing image: {name}. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()