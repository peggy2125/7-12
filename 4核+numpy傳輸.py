import cv2
import numpy as np
import os
import time
from multiprocessing import Process, Queue, shared_memory
from queue import Empty
from threading import Thread


def load_image(image_path):
    image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.tobytes(), image.shape, os.path.basename(image_path)

def thread2(input_queue, output_queue):
    while True:
        image_path = input_queue.get()
        if image_path is None:
            output_queue.put(None)  # 发送结束信号
            break
        imagetobytes, shape, filename = load_image(image_path)
        output_queue.put((imagetobytes,shape, filename))

def gaussianandbgsub(image_data, shape, blurred_bg):
    image = np.frombuffer(image_data, dtype=np.uint8).reshape(shape)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    bg_sub = cv2.subtract(blurred_bg, blurred)
    return bg_sub.tobytes()

def binary(bg_sub_data, shape):
    bg_sub= np.frombuffer(bg_sub_data, dtype=np.uint8).reshape(shape)
    _, binary_image = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)
    return binary_image.tobytes()

def morphological_operations(binary_image_data, shape):
    kernel =np.ones((3, 3), np.uint8)  # 使用NumPy创建kernel    
    binary_image = np.frombuffer(binary_image_data, dtype=np.uint8).reshape(shape)
    dilate1 = cv2.dilate(binary_image, kernel, iterations=2)
    erode1 = cv2.erode(dilate1, kernel, iterations=2)
    erode2 = cv2.erode(erode1, kernel, iterations=1)
    return erode2.tobytes()

def find_contours(processed_image_data, shape):
    processed_image = np.frombuffer(processed_image_data, dtype=np.uint8).reshape(shape)
    edges = cv2.Canny(processed_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(processed_image)
    cv2.drawContours(contour_image, contours, -1, (255), 1)
    return contour_image.tobytes()
 
def worker_gusandbgsub(input_queue, output_queue, bg_shm_name, bg_shape):
    existing_shm = shared_memory.SharedMemory(name=bg_shm_name)
    blurred_bg = np.ndarray(bg_shape, dtype=np.uint8, buffer=existing_shm.buf)
    while True:
        try:
            item = input_queue.get(timeout=1)
            if item is None:
                break
            imagetobytes,shape,filename=item
            bg_sub_data = gaussianandbgsub(imagetobytes, shape, blurred_bg)
            output_queue.put((bg_sub_data, shape, filename))
        except Empty:
            continue
        except Exception as e:
            print(f"Error in worker_gusandbgsub: {e}")
    output_queue.put(None)
    
def worker_binary(input_queue, output_queue):
    while True:
        try:
            item = input_queue.get(timeout=1)
            if item is None:
                break
            bg_sub_data, shape,filename = item
            binaryimg_data = binary(bg_sub_data, shape)
            output_queue.put((binaryimg_data, shape, filename))
        except Empty:
            continue
        except Exception as e:
            print(f"Error in worker_binary: {e}")
    output_queue.put(None)


def worker_morphological(input_queue, output_queue):
    while True:    
        try:
            item = input_queue.get(timeout=1)
            if item is None:
                break
            binaryimg_data, shape, filename = item
            processedimg_data = morphological_operations(binaryimg_data, shape)
            output_queue.put((processedimg_data, shape, filename))
        except Empty:
            continue
        except Exception as e:
            print(f"Error in worker_morphological: {e}")
    output_queue.put(None)
        


def worker_contour(input_queue, output_queue):
    while True:
        try:
            item = input_queue.get(timeout=1)
            if item is None:
                break
            processedimg_data, shape, filename = item
            contour_image_data = find_contours(processedimg_data, shape)
            output_queue.put((contour_image_data, shape, filename))
        except Empty:
            continue
        except Exception as e:
            print(f"Error in worker_morphological: {e}")
        output_queue.put(None)


def main():
    directory = r''
    background_path = os.path.join(directory, 'background.tiff')
    files = [f for f in os.listdir(directory) if f.endswith('.tiff') and f != 'background.tiff']
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
        # 开始处理图像
    
    #創建共用記憶體
    shm = shared_memory.SharedMemory(create=True, size=blurred_bg.nbytes)
    shared_bg = np.ndarray(blurred_bg.shape, dtype=blurred_bg.dtype, buffer=shm.buf)
    np.copyto(shared_bg, blurred_bg)
    
    queue4 = Queue()
    rawimage_queue=Queue()
    for image in files:
            queue4.put(os.path.join(directory, image))
    queue4.put(None)
    
    t1 = Thread(target=thread2, args=(queue4, rawimage_queue))
    
    
    
    queue2 = Queue()  #queue2 = JoinableQueue()
    queue3 = Queue()   # queue3 = JoinableQueue()
    queue6 = Queue()   # queue6 = JoinableQueue()
    result_queue = Queue()
    
    processes = [
        Process(target=worker_gusandbgsub, args=(rawimage_queue, queue2, shm.name, background.shape)),
        Process(target=worker_binary, args=(queue2, queue3)),
        Process(target=worker_morphological, args=(queue3, queue6)),
        Process(target=worker_contour, args=(queue6, result_queue))
    ]
    
    start_time = time.time()
    try:
        t1.start()
        
        t1.join()
        rawimage_queue.put(None)
        
        for p in processes:
            p.start()
        
        # 开始收集结果
        results = {}
        result_count = 0
        empty_count = 0
        max_empty_count = 10  # 允许连续空队列的最大次数

        while result_count < len(files) and empty_count < max_empty_count:
            try:
                item = result_queue.get(timeout=1)
                if item is None:
                    empty_count += 1
                    continue
                contour_image_data, shape, filename = item
                contour_image= np.frombuffer(contour_image_data, dtype=np.uint8).reshape(shape)
                results[filename] = contour_image
                result_count += 1
                empty_count = 0  # 重置空计数
            except Empty:
                empty_count += 1
        
        for q in [rawimage_queue, queue2, queue3]:
            q.put(None)
        
        for p in processes:
            p.join(timeout=5)
            
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        print(f"Average time per image: {(end_time-start_time)/len(files):.6f} sec")

        for image_name, contour_image in results.items():
            cv2.imshow(f'Processed Image: {image_name}', contour_image)
            print(f"Showing image: {image_name}. Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()