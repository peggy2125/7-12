import cv2
import numpy as np
import os
import time
from multiprocessing import Process, Queue, JoinableQueue,shared_memory
from queue import Empty
from concurrent.futures import ThreadPoolExecutor

BATCH_SIZE = 5

# 加载图像函数
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), os.path.basename(image_path)
# 图像处理函数
def process_image(image, blurred_bg):
    # 图像进行高斯模糊
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # 背景减除
    bg_sub = cv2.subtract(blurred_bg, blurred)
    # 二值化
    _, binary = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)
    # 形态学操作
    kernel =np.ones((3, 3), np.uint8)  # 使用NumPy创建kernel
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilate1 = cv2.dilate(binary, kernel, iterations=2)
    erode1 = cv2.erode(dilate1, kernel, iterations=2)
    erode2 = cv2.erode(erode1, kernel, iterations=1)
    return erode2

# 轮廓检测函数
def find_contours(processed_image):
    edges = cv2.Canny(processed_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(processed_image)
    cv2.drawContours(contour_image, contours, -1, (255), 1)
    return contour_image

# 图像加载工作进程
def worker_load(input_queue, output_queue):
    with ThreadPoolExecutor() as executor:
        batch = []
        while True:
            image_path = input_queue.get(timeout=5)
            if image_path is None:
                #if batch:
                #    output_queue.put(batch)
                output_queue.put(None)
                break
            future = executor.submit(load_image, image_path)
            batch.append(future.result())
            if len(batch) == BATCH_SIZE:
                output_queue.put(batch)
                batch = []
            input_queue.task_done()

# 图像处理工作进程
def worker_process(input_queue, output_queue, bg_shm_name, bg_shape):
    existing_shm = shared_memory.SharedMemory(name=bg_shm_name)
    blurred_bg = np.ndarray(bg_shape, dtype=np.uint8, buffer=existing_shm.buf)
    try:
        while True:
            try:
                batch = input_queue.get(timeout=5)
                if batch is None:
                    output_queue.put(None)
                    break
                processed_batch = []
                for image, image_name in batch:
                    processed = process_image(image, blurred_bg)
                    processed_batch.append((processed, image_name))
                output_queue.put(processed_batch)
                input_queue.task_done()
            except Empty:
                continue
    finally:
        existing_shm.close()
    

def worker_contour(input_queue, output_queue):
    while True:
        try:
            batch = input_queue.get(timeout=5)
            if batch is None:
                output_queue.put(None)
                break
            contour_batch = []
            for processed, image_name in batch:
                contour_image = find_contours(processed)
                if contour_image is not None:
                    contour_batch.append((contour_image, image_name))
            output_queue.put(contour_batch)
            input_queue.task_done()
        except Empty:
            continue

# 主函数
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
    queue1 = JoinableQueue()
    queue2 = JoinableQueue()
    queue3 = JoinableQueue()
    result_queue = Queue()

    process1 = Process(target=worker_load, args=(queue1, queue2))
    process2 = Process(target=worker_process, args=(queue2, queue3, shm.name, blurred_bg.shape))
    process3 = Process(target=worker_contour, args=(queue3, result_queue))
    #print(f"Creating process2 with args: queue2, queue3, {shm.name}, {blurred_bg.shape}")
    processes = [process1, process2, process3]

    start_time = time.time()

    try:
        # 启动所有进程
        for p in processes:
            p.start()

        # 将图像文件路径放入队列
        for image in files:
            queue1.put(os.path.join(directory, image))

        # 发送结束信号
        for _ in processes:
            queue1.put(None)

        # 收集处理结果
        results = {}
        while True:
            try:
                batch = result_queue.get(timeout=10)
                if batch is None:
                    break
                for contour_image, image_name in batch:
                    results[image_name] = contour_image
            except Empty:
                break
        # 等待所有进程结束
        for p in processes:
            p.join(timeout=30)
            if p.is_alive():
                print(f"Process {p.name} did not terminate in time")

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
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        shm.close()
        shm.unlink()
if __name__ == "__main__":
    main()
