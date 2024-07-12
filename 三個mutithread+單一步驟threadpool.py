import cv2
import numpy as np
import os
import time
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

def load_images(directory):
    images_list = {}
    for filename in os.listdir(directory):
        if filename.endswith('.tiff') and filename != 'background.tiff':
            path = os.path.join(directory, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            images_list[filename] = {
                'name': filename,
                'path': path,
                'image': image
            }
    return images_list


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


def process_images(images,background):
    # Create kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Apply processing steps
    blurred = apply_gaussian_blur(images)
    bg_sub = subtract_background(blurred, background)
    binary = apply_threshold(bg_sub)
    dilate1 = apply_morphology(binary, cv2.dilate, kernel, 2)
    erode1 = apply_morphology(dilate1, cv2.erode, kernel, 2)
    erode2 = apply_morphology(erode1, cv2.erode, kernel, 1)
    #dilate2 = apply_morphology(erode2, cv2.dilate, kernel, 1)
    edges = apply_canny(erode2)
    return edges


def find_contours(processed_image):
    edges = cv2.Canny(processed_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(processed_image)
    cv2.drawContours(contour_image, contours, -1, (255), 1)
    return contour_image

        
def worker_load(directory, output_queue):
    images_list = load_images(directory)
    for filename, image_data in images_list.items():
        output_queue.put((image_data['image'], filename))
    output_queue.put(None)


def worker_process(input_queue, output_queue, background):
    while True:
        item = input_queue.get()
        if item is None:
            output_queue.put(None)
            break
        image, filename = item
        processed = process_images({filename: image}, background)
        output_queue.put((processed[filename], filename))
        input_queue.task_done()

# 修改: 更新 worker_contour 函數
def worker_contour(input_queue, output_queue):
    while True:
        item = input_queue.get()
        if item is None:
            output_queue.put(None)
            break
        processed, filename = item
        contour_image = find_contours(processed)
        output_queue.put((contour_image, filename))
        input_queue.task_done()


def main():
    directory = 'Test_images/Slight under focus'
    background_path = os.path.join(directory, 'background.tiff')
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    background = cv2.GaussianBlur(background, (5, 5), 0)
    files = [f for f in os.listdir(directory) if f.endswith('.tiff') and f != 'background.tiff']
    
    start_time=time.time()
    #for image in files:
    #    queue1.put(os.path.join(directory, image))
    
    #queue1 = Queue()
    queue2 = Queue()
    queue3 = Queue()
    result_queue = Queue()

    thread1 = Thread(target=worker_load, args=(directory, queue2))
    thread2 = Thread(target=worker_process, args=(queue2, queue3, background))
    thread3 = Thread(target=worker_contour, args=(queue3, result_queue))

    thread1.start()
    thread2.start()
    thread3.start()

    

    #queue1.join()
    queue2.join()
    queue3.join()

    #queue1.put(None)
    thread1.join()
    thread2.join()
    thread3.join()

    results = {}
    while not result_queue.empty():
        item = result_queue.get()
        if item is None:
            break
        contour_image, image_name = item
        results[image_name] = contour_image
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Average time per image: {(end_time - start_time) / len(files):.6f} seconds")
    for image in files:
        image_name = os.path.basename(image)
        if image_name in results:
            cv2.imshow(f'Processed Image: {image_name}', results[image_name])
            print(f"Showing image: {image_name}. Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()