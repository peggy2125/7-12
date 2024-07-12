import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

#def background_process(background_path)
#    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
#    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
#    return background,blurred_bg
    
def read_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
def trace_contours(edge_image):
    contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return [contour.reshape(-1, 2) for contour in contours]

    
def preprocess_image(image, blurred_bg):
    with ThreadPoolExecutor() as executor:
        blurred_future=executor.submit(cv2.GaussianBlur, image)
        blurred=blurred_future.result()
        bg_sub_future = executor.submit(cv2.subtract,blurred_bg, blurred)
        bg_sub=bg_sub_future.result()
        binary_future = executor.submit(cv2.threshold,bg_sub, 10, 255, cv2.THRESH_BINARY)
        _, binary=binary_future.result()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilate1_future=executor.submit(cv2.dilate,binary, kernel, iterations=2)
        dilate1=dilate1_future.result()
        erode1_future=executor.submit(cv2.erode,dilate1, kernel, iterations=2)
        erode1=erode1_future.result()
        erode2_future=executor.submit(cv2.erode,erode1, kernel, iterations=1)
        erode2=erode2_future.result() 
        dilate2_future=executor.submit(cv2.dilate,erode2, kernel, iterations=1)
        dilate2=dilate2_future.result()
        return dilate2

def edge_detection(img):
        with ThreadPoolExecutor() as executor:
            edge_future=executor.submit(cv2.Canny(img, 50, 150))
            edge = edge_future.result()
            contours_future = executor.submit(trace_contours(edge))
            contours = contours_future.result()
            return contours
    
def process_single_image(image_path, blurred_bg):
    with ProcessPoolExecutor(max_workers=1) as read_executor:
        image_future= read_executor.submit(read_image,image_path)
        image=image_future.result()
        
    with ProcessPoolExecutor(max_workers=1) as preprocess_executor:
        preprocessimage_future= preprocess_executor.submit(preprocess_image, image, blurred_bg)
        preprocessimage = preprocessimage_future.result()
    
    with ProcessPoolExecutor(max_workers=1) as edge_executor:
        processimage_future=edge_executor.submit(edge_detection, preprocessimage)
        processimage=processimage_future.result()
    
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, processimage, -1, (255), 1)
    cv2.imshow('Processed Image', contour_image)  #顯示結果圖像在processed image中
    cv2.waitKey(0)  #使程序暫停,等待用戶按下任意鍵，可以確保圖像窗口一直保持打開,直到用戶手動關閉它
    cv2.destroyAllWindows()

    #while True:
    #    for name, img in images_dict.items():      
     #       cv2.imshow(name, img)
        
      #  key = cv2.waitKey(1)
      #  if key!=-1:          #按下任意鍵
       #     break

    #cv2.destroyAllWindows()


directory = 'Test_images/Slight under focus'
background_path = os.path.join(directory, 'background.tiff')
files = [f for f in os.listdir(directory) if f.endswith('.tiff') and f != 'background.tiff']
num_images = len(files)
    
    # 預處理背景圖片（只需要執行一次）
background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
    
start_time = time.time()
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_single_image, os.path.join(directory, image), blurred_bg) for image in files]
        
    for future in as_completed(futures):
     # 這裡可以添加額外的處理邏輯，如果需要的話
        pass
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
print(f"Average time per image: {(end_time - start_time) / num_images:.2f} seconds")
    
cv2.waitKey(0)
cv2.destroyAllWindows()

