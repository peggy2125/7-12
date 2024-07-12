import cv2
import numpy as np
import os
import time
from multiprocessing import Process, Queue, shared_memory
from queue import Empty

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.tobytes(), image.shape, os.path.basename(image_path)

def process_image(image_data, shape, blurred_bg):
    image = np.frombuffer(image_data, dtype=np.uint8).reshape(shape)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    bg_sub = cv2.subtract(blurred_bg, blurred)
    _, binary = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilate1 = cv2.dilate(binary, kernel, iterations=2)
    erode1 = cv2.erode(dilate1, kernel, iterations=2)
    erode2 = cv2.erode(erode1, kernel, iterations=1)
    return erode2.tobytes()

def find_contours(processed_data, shape):
    processed_image = np.frombuffer(processed_data, dtype=np.uint8).reshape(shape)
    edges = cv2.Canny(processed_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(processed_image)
    cv2.drawContours(contour_image, contours, -1, (255), 1)
    return contour_image.tobytes()

def worker_load(input_queue, output_queue):
    while True:
        try:
            image_path = input_queue.get(timeout=1)
            if image_path is None:
                break
            image_data, shape, image_name = load_image(image_path)
            output_queue.put((image_data, shape, image_name))
        except Empty:
            continue
    output_queue.put(None)
    
def worker_process(input_queue, output_queue, bg_shm_name, bg_shape):
    existing_shm = shared_memory.SharedMemory(name=bg_shm_name)
    blurred_bg = np.ndarray(bg_shape, dtype=np.uint8, buffer=existing_shm.buf)
    while True:
        try:
            item = input_queue.get(timeout=1)
            if item is None:
                break
            image_data, shape, image_name = item
            processed_data = process_image(image_data, shape, blurred_bg)
            output_queue.put((processed_data, shape, image_name))
        except Empty:
            continue
    existing_shm.close()
    output_queue.put(None)

def worker_contour(input_queue, output_queue):
    while True:
        try:
            item = input_queue.get(timeout=1)
            if item is None:
                break
            processed_data, shape, image_name = item
            contour_data = find_contours(processed_data, shape)
            output_queue.put((contour_data, shape, image_name))
        except Empty:
            continue
    output_queue.put(None)

def main():
    directory = 'Test_images/Slight under focus'
    background_path = os.path.join(directory, 'background.tiff')
    files = [f for f in os.listdir(directory) if f.endswith('.tiff') and f != 'background.tiff']
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
    #print(f"Background shape: {background.shape}")
    #print(f"Background size: {background.nbytes} bytes")

    # 创建共享内存
    shm = shared_memory.SharedMemory(create=True, size=blurred_bg.nbytes)
    shared_bg = np.ndarray(blurred_bg.shape, dtype=blurred_bg.dtype, buffer=shm.buf)
    np.copyto(shared_bg, blurred_bg)

    queue1 = Queue()
    queue2 = Queue()
    queue3 = Queue()
    result_queue = Queue()

    process1 = Process(target=worker_load, args=(queue1, queue2))
    process2 = Process(target=worker_process, args=(queue2, queue3, shm.name, background.shape))
    process3 = Process(target=worker_contour, args=(queue3, result_queue))

    processes = [process1, process2, process3]

    start_time = time.time()

    try:
        for p in processes:
            p.start()

        for image in files:
            queue1.put(os.path.join(directory, image))

        queue1.put(None)

        results = {}
        result_count = 0

        while result_count < len(files):
            try:
                item = result_queue.get(timeout=30)  # 增加超時時間
                if item is None:
                    break
                contour_data, shape, image_name = item
                contour_image = np.frombuffer(contour_data, dtype=np.uint8).reshape(shape)
                results[image_name] = contour_image
                result_count += 1
                #print(f"Processed {result_count}/{len(files)}")  # 添加進度打印
            except Empty:
                print("Timeout while waiting for results")
                break

        for p in processes:
            p.join(timeout=10)

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        print(f"Average time per image: {(end_time-start_time)/len(files):.6f} sec")
        
        for image_name, contour_image in results.items():
            cv2.imshow(f'Processed Image: {image_name}', contour_image)
            print(f"Showing image: {image_name}. Press any key to continue...")
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred in main: {e}")
    finally:
        shm.close()
        shm.unlink()
	
if __name__ == "__main__":
    main()
