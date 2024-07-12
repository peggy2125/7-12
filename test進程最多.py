import cv2
import numpy as np
import os
import time
import logging
from multiprocessing import Process, Queue, JoinableQueue, shared_memory, cpu_count, Event
from queue import Empty
from concurrent.futures import ThreadPoolExecutor

def load_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image.tobytes(), image.shape, os.path.basename(image_path)
    except Exception as e:
        return None

def gaussian_blur(image_data, shape):
    try:
        image = np.frombuffer(image_data, dtype=np.uint8).reshape(shape)
        return cv2.GaussianBlur(image, (5, 5), 0).tobytes()
    except Exception as e:
        return None

def background_subtraction(image_data, shape, blurred_bg):
    try:
        image = np.frombuffer(image_data, dtype=np.uint8).reshape(shape)
        return cv2.subtract(blurred_bg, image).tobytes()
    except Exception as e:
        return None

def thresholding(image_data, shape):
    try:
        image = np.frombuffer(image_data, dtype=np.uint8).reshape(shape)
        _, binary = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
        return binary.tobytes()
    except Exception as e:
        return None

def morphological_operations(image_data, shape):
    try:
        image = np.frombuffer(image_data, dtype=np.uint8).reshape(shape)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilate1 = cv2.dilate(image, kernel, iterations=2)
        erode1 = cv2.erode(dilate1, kernel, iterations=2)
        erode2 = cv2.erode(erode1, kernel, iterations=1)
        return erode2.tobytes()
    except Exception as e:
        return None

def find_contours(image_data, shape):
    try:
        image = np.frombuffer(image_data, dtype=np.uint8).reshape(shape)
        edges = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, contours, -1, (255), 1)
        return contour_image.tobytes()
    except Exception as e:
        return None

def worker(input_queue, output_queue, operation, bg_shm_name=None, bg_shape=None, done_event=None):
    if bg_shm_name and bg_shape:
        try:
            existing_shm = shared_memory.SharedMemory(name=bg_shm_name)
            blurred_bg = np.ndarray(bg_shape, dtype=np.uint8, buffer=existing_shm.buf)
        except Exception as e:
            return
    else:
        blurred_bg = None

    while not done_event.is_set():
        try:
            item = input_queue.get(timeout=1)
            if item is None:
                break
            if operation == 'load':
                result = load_image(item)
            else:
                image_data, shape, image_name = item
                if operation == 'blur':
                    result = gaussian_blur(image_data, shape), shape, image_name
                elif operation == 'subtract':
                    result = background_subtraction(image_data, shape, blurred_bg), shape, image_name
                elif operation == 'threshold':
                    result = thresholding(image_data, shape), shape, image_name
                elif operation == 'morphology':
                    result = morphological_operations(image_data, shape), shape, image_name
                elif operation == 'contour':
                    result = find_contours(image_data, shape), shape, image_name
                else:
                    raise ValueError(f"Unknown operation: {operation}")
            
            if result is not None:
                output_queue.put(result)
            input_queue.task_done()
        except Empty:
            continue
        except Exception as e:
            pass

    if bg_shm_name:
        existing_shm.close()
    output_queue.put(None)

def main():
    # 设置目录和文件路径
    directory = 'Test_images/Slight under focus'
    background_path = os.path.join(directory, 'background.tiff')
    files = [f for f in os.listdir(directory) if f.endswith('.tiff') and f != 'background.tiff']
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
    
    shm = shared_memory.SharedMemory(create=True, size=blurred_bg.nbytes)
    shared_bg = np.ndarray(blurred_bg.shape, dtype=blurred_bg.dtype, buffer=shm.buf)
    np.copyto(shared_bg, blurred_bg)
    
    # 创建队列和进程
    num_processes = min(cpu_count(), 6)  # 使用最多6个进程，避免过度并行
    queues = [JoinableQueue() for _ in range(num_processes + 1)]
    result_queue = queues[-1]

    operations = ['load', 'blur', 'subtract', 'threshold', 'morphology', 'contour']
    processes = []
    done_event = Event()
    for i, operation in enumerate(operations):
        if operation == 'subtract':
            p = Process(target=worker, args=(queues[i], queues[i+1], operation, shm.name, blurred_bg.shape, done_event))
        else:
            p = Process(target=worker, args=(queues[i], queues[i+1], operation, None, None, done_event))
        processes.append(p)

    start_time = time.time()

    try:
        # 启动所有进程
        for p in processes:
            p.start()

        # 将图像文件路径放入队列
        for image in files:
            queues[0].put(os.path.join(directory, image))

        # 发送结束信号
        for _ in range(len(processes)):
            queues[0].put(None)

        # 收集处理结果
        results = {}
        result_count = 0
        none_count = 0
        while result_count < len(files) and none_count < len(processes):
            try:
                item = result_queue.get(timeout=10)
                if item is None:
                    none_count += 1
                    continue
                contour_image_data, shape, image_name = item
                contour_image = np.frombuffer(contour_image_data, dtype=np.uint8).reshape(shape)
                results[image_name] = contour_image
                result_count += 1
                #print(f"Processed image {result_count}/{len(files)}: {image_name}")
            except Empty:
                print("Timeout while waiting for results")
                break
            except Exception as e:
                print(f"Error processing result: {e}")

        # 设置结束事件
        done_event.set()

        # 等待所有进程结束
        for p in processes:
            p.join(timeout=30)
            if p.is_alive():
                print(f"Process {p.name} did not terminate in time")
                p.terminate()

        # 计算并打印执行时间
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        print(f"Average time per image: {(end_time-start_time)/len(files):.6f} sec")
        
        # 显示处理后的图像
        for image_name, contour_image in results.items():
            cv2.imshow(f'Processed Image: {image_name}', contour_image)
            print(f"Showing image: {image_name}. Press any key to continue...")
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred in main: {e}")
    finally:
        # 确保所有进程都被终止
        done_event.set()
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        shm.close()
        shm.unlink()

    #print("Main function finished")

if __name__ == "__main__":
    main()
