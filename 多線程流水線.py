import cv2
import numpy as np
import os
import time
from queue import Queue
from threading import Thread

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
def gaussianblur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)
def background_sub(blurred_bg, blurred):
    return cv2.subtract(blurred_bg, blurred)
def binary(bg_sub):
    _, binary_im = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)
    return binary_im
def morphology(binaryim):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilate1 = cv2.dilate(binaryim, kernel, iterations=2)
    erode1 = cv2.erode(dilate1, kernel, iterations=2)
    erode2 = cv2.erode(erode1, kernel, iterations=1)
    #dilate2 = cv2.dilate(erode2, kernel, iterations=1)
    return erode2
def canny(processed_image):
    edges = cv2.Canny(processed_image, 50, 150)
    return edges
def findcontours(edges):    
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours
def drawcontours(contours, processed):
    contour_image = np.zeros_like(processed)
    cv2.drawContours(contour_image, contours, -1, (255), 1)
    return contour_image

def worker(input_queue, output_queue, stage, blurred_bg=None):
    while True:
        item = input_queue.get()
        if item is None:
            break
        if stage == 'load':
            image = load_image(item)
            output_queue.put((image, os.path.basename(item)))
        elif stage == 'gaussian':
            image, image_name = item
            blurred = gaussianblur(image)
            output_queue.put((blurred, image_name))
        elif stage == 'bdsub':
            blurred, image_name = item
            bg_sub = background_sub(blurred_bg, blurred)
            output_queue.put((bg_sub, image_name))
        elif stage == 'binary':
            bg_sub, image_name = item
            binaryimg = binary(bg_sub)
            output_queue.put((binaryimg, image_name))
        elif stage == 'morphology':
            binaryimg, image_name = item
            processed = morphology(binaryimg)
            output_queue.put((processed, image_name))
        elif stage == 'canny':
            processed, image_name= item
            edge = canny(processed)
            output_queue.put((edge, image_name,processed))
        elif stage == 'findcontours':
            edge, image_name,processed = item
            contour = findcontours(edge)
            output_queue.put((contour, image_name,processed))
        elif stage == 'drawcontour':
            contour, image_name,processed = item
            contour_image = drawcontours(contour, processed)
            output_queue.put((contour_image, image_name))
    output_queue.put(None)

def main():
    directory = 'Test_images/Slight under focus'
    background_path = os.path.join(directory, 'background.tiff')
    files = [f for f in os.listdir(directory) if f.endswith('.tiff') and f != 'background.tiff']
    blurred_bg = cv2.GaussianBlur(load_image(background_path), (5, 5), 0)
    
    queues = [Queue() for _ in range(9)]

    threads = [
        Thread(target=worker, args=(queues[0], queues[1], 'load')),
        Thread(target=worker, args=(queues[1], queues[2], 'gaussian')),
        Thread(target=worker, args=(queues[2], queues[3], 'bdsub', blurred_bg)),
        Thread(target=worker, args=(queues[3], queues[4], 'binary')),
        Thread(target=worker, args=(queues[4], queues[5], 'morphology')),
        Thread(target=worker, args=(queues[5], queues[6], 'canny')),
        Thread(target=worker, args=(queues[6], queues[7], 'findcontours')),
        Thread(target=worker, args=(queues[7], queues[8], 'drawcontour')),
    ]

    for thread in threads:
        thread.start()

    start_time = time.time()

    for image in files:
        queues[0].put(os.path.join(directory, image))
    queues[0].put(None)
    
    for thread in threads:
        thread.join()

    results = {}
    while True:
        item = queues[8].get()
        if item is None:
            break
        contour_image, image_name = item
        results[image_name] = contour_image

    end_time = time.time()    
    print(f"Total execution time: {end_time - start_time:.6f} seconds")
    print(f"Average time per image: {(end_time - start_time) / len(files):.6f} seconds")

    for image_name, contour_image in results.items():
        cv2.imshow(f'Processed Image: {image_name}', contour_image)
        print(f"Showing image: {image_name}. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

