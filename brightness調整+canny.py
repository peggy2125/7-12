import cv2
import numpy as np
import time
import os


def trace_contours(edge_image):
    # Placeholder for where you would implement contour tracing in FPGA
    # This function simulates tracing edges by finding non-zero pixels
    contours = []
    visited = np.zeros_like(edge_image, dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connectivity

    for i in range(edge_image.shape[0]):
        for j in range(edge_image.shape[1]):
            if edge_image[i, j] != 0 and not visited[i, j]:
                contour = []
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if not visited[x, y]:
                        visited[x, y] = True
                        contour.append((x, y))
                        for dx, dy in directions:
                            xn, yn = x + dx, y + dy
                            if 0 <= xn < edge_image.shape[0] and 0 <= yn < edge_image.shape[1]:
                                if edge_image[xn, yn] != 0 and not visited[xn, yn]:
                                    stack.append((xn, yn))
                if contour:
                    contours.append(contour)
    return contours


def process_image(image_path, background_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mean_brightness = np.mean(image)
    print(f"The average brightness of the input image is: {mean_brightness}")
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
    cv2.imshow('raw', image)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    start_time = time.time()

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imshow('blurred', blurred)

    # Background subtraction
    print(blurred.shape, blurred_bg.shape)
    bg_sub = cv2.subtract(blurred_bg, blurred)
    cv2.imshow('bg_sub', bg_sub)

    # Apply threshold
    _, binary = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)
    # binary = cv2.adaptiveThreshold(bg_sub, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
    cv2.imshow('binary', binary)

    # Erode and dilate to remove noise
    dilate1 = cv2.dilate(binary, kernel, iterations = 2)
    cv2.imshow('dilate1', dilate1)
    erode1 = cv2.erode(dilate1, kernel, iterations = 2)
    cv2.imshow('erode1', erode1)
    erode2 = cv2.erode(erode1, kernel, iterations = 1)
    cv2.imshow('erode2', erode2)
    dilate2 = cv2.dilate(erode2, kernel, iterations = 1)
    cv2.imshow('dilate2', dilate2)



    # Apply Canny edge detector to find edges
    edges = cv2.Canny(erode2, 50, 150)
    cv2.imshow('canny edges', edges)

    # Trace contours from the edge image
    contours = trace_contours(edges)

    end_time = time.time()
    dif_time = end_time - start_time
    print(dif_time)

    # Prepare an image to draw the contours
    contour_image = np.zeros_like(image)

    # Draw each contour
    for contour in contours:
        for x, y in contour:
            contour_image[x, y] = 255

    # Show the resulting image
    cv2.imshow('Processed Image', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Replace 'path_to_image.tif' with your image file path
process_image('Test_images/Slight under focus/0066_bright.tif', 'Test_images/Slight under focus/background.tiff')

# Set the directory containing your files
# directory = 'Test_images/Slight under focus'
# # Get a list of all tiff files
# files = [f for f in os.listdir(directory) if f.endswith('.tiff')]
# for image in files:
#     image_path = os.path.join(directory, image)
#     print(image_path)
#     process_image(image_path, 'Test_images/Slight under focus/background.tiff')