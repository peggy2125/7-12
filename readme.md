**RT-DC sample code   version : python 3.9.19 & opencv 4.10.0**

1. **original code with note** (Canny_edge 註解.py)
```python
import cv2
import numpy as np
import time
import os

def trace_contours(edge_image): #edge image=一個二值化或邊緣檢測後的圖像
    # Placeholder for where you would implement contour tracing in FPGA
    # This function simulates tracing edges by finding non-zero pixels
    contours = [] #創建空陣列，用於儲存已找到的輪廓
    visited = np.zeros_like(edge_image, dtype=bool)  #創建了一個與 edge_image 大小相同的bool類型的 NumPy 數組,用於記錄已經訪問過的像素
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connectivity  #,表示從當前像素出發的 4 個方向(上、下、左、右)。這個列表將用於遍歷圖像,尋找連續的非零像素。

    for i in range(edge_image.shape[0]):  #重複edge_image 陣列的行(橫)數次
        for j in range(edge_image.shape[1]):   #重複edge_image 陣列的列(直)數次
            if edge_image[i, j] != 0 and not visited[i, j]:  #如果像素還沒被檢查且非為0時執行
                contour = []   #另一個空陣列(儲存用)
                stack = [(i, j)]   #將當前的位置 (i, j) 加入到堆疊中,作為輪廓追蹤的起點
                while stack:   #迴圈直到堆疊empty
                    x, y = stack.pop()  #從堆疊中彈出一個位置 (x, y)
                    if not visited[x, y]:  #如果x,y還未被訪問
                        visited[x, y] = True    #將該位置標記為已訪問
                        contour.append((x, y))    #將該位置加入到當前輪廓的點列表中
                        for dx, dy in directions:   #這個 for 迴圈遍歷 4 個方向(directions列表)
                            xn, yn = x + dx, y + dy    #計算當前位置在某個方向上的相鄰位置(微小變化)
                            if 0 <= xn < edge_image.shape[0] and 0 <= yn < edge_image.shape[1]:   #若相鄰位置大於0且小於範圍內(是否在圖像範圍中)
                                if edge_image[xn, yn] != 0 and not visited[xn, yn]:   #如果相鄰位置同樣未被訪問
                                    stack.append((xn, yn))   #將該位置標記為已訪問
                if contour:  #檢查當前輪廓是否包含任何點
                    contours.append(contour)    #如果當前輪廓不為空,則將其加入到最終的 contours 列表中
    return contours  #回傳最終contours列表


def process_image(image_path, background_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)   #使用 OpenCV 函數 cv2.imread() 以灰度模式讀取圖像,並將其存儲在image變數中
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)   #使用 OpenCV 函數 cv2.imread() 以灰度模式讀取背景圖像,並將其存儲在background變數中。
    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)  #使用 OpenCV 函數 cv2.GaussianBlur() 對"背景圖像"進行高斯模糊處理(低通濾波，降低noise與柔化),以平滑背景細節。(5, 5)是高斯核(模糊的程度,值越大,模糊效果越強)的大小,0是標準差。
    cv2.imshow('raw', image)  #使用 OpenCV 函數 cv2.imshow() 在一個名為 'raw' 的窗口中顯示原始圖像
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))   #使用 OpenCV 函數 cv2.getStructuringElement() 創建一個 3x3 的十字形結構元素,用於後續的形態學操作。

    start_time = time.time()  #將值儲存在start time中用以計算時間長

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  #使用 OpenCV 的 cv2.GaussianBlur() 函數對image變量進行高斯模糊處理
    cv2.imshow('blurred', blurred)  #使用 OpenCV 的 cv2.imshow() 函數在一個名為 'blurred' 的窗口中顯示模糊處理後的圖像

    # Background subtraction 去背
    print(blurred.shape, blurred_bg.shape)  #顯示了模糊後的圖像blurred和模糊背景blurred_bg的維度,用於確認兩者的尺寸是否一致
    bg_sub = cv2.subtract(blurred_bg, blurred)  #使用 OpenCV 的 cv2.subtract() 函數從模糊背景blurred_bg中減去模糊後的圖像blurred。這個操作可以突出圖像中的前景區域,因為前景區域與背景的差異會被保留下來。
    cv2.imshow('bg_sub', bg_sub)   #使用 OpenCV 的 cv2.imshow() 函數在一個名為 'bg_sub' 的窗口中顯示背景減除的結果

    # Apply threshold 設定閾值，二極化處理(大於或小於閾值為黑或白)
    _, binary = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)  #cv2.threshold() 函數用於對輸入圖像進行二值化處理，10設置了二值化的閾值(像素值大於 10 時,該像素將被設置為 255(白色),否則設置為 0(黑色)
    # binary = cv2.adaptiveThreshold(bg_sub, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
    cv2.imshow('binary', binary)    #使用 OpenCV 的 cv2.imshow() 函數在一個名為 'binary' 的窗口中顯示二極化的結果

    # Erode and dilate to remove noise
    dilate1 = cv2.dilate(binary, kernel, iterations = 2)  # cv2.dilate() 函數對二值化後的圖像 binary 進行膨脹操作，kernel 定義膨脹的模式，iterations = 2 指定了膨脹操作重複進行的次數，總體可填補一些小的洞洞,使前景物體更加連續
    cv2.imshow('dilate1', dilate1)   #使用 OpenCV 的 cv2.imshow() 函數在一個名為 'dilate1' 的窗口中顯示膨脹的結果
    erode1 = cv2.erode(dilate1, kernel, iterations = 2)  #使用 cv2.erode() 函數對膨脹後的圖像 dilate1 進行腐蝕操作，去除一些小的雜訊
    cv2.imshow('erode1', erode1)
    erode2 = cv2.erode(erode1, kernel, iterations = 1)  #再次對 erode1 進行腐蝕操作,以進一步去除雜訊
    cv2.imshow('erode2', erode2)
    dilate2 = cv2.dilate(erode2, kernel, iterations = 1)  #最後,對 erode2 進行一次膨脹操作,以彌補之前腐蝕造成的一些細節損失
    cv2.imshow('dilate2', dilate2)  



    # Apply Canny edge detector to find edges
    edges = cv2.Canny(erode2, 50, 150)  #使用了 OpenCV 的 cv2.Canny() 函數對圖像 erode2 進行Canny邊緣檢測，50 是低閾值,小於該值的邊緣將被抑制。150 是高閾值,大於該值的邊緣將被保留。
    cv2.imshow('canny edges', edges)  #白色部分表示檢測到的邊緣,黑色部分表示非邊緣區域

    # Trace contours from the edge image    對canny進行輪廓追蹤
    contours = trace_contours(edges)  #使用了一個名為 trace_contours() 的函數(自訂義),並將 edges 圖像傳遞進去。這個函數用來檢測和跟蹤圖像中的輪廓

    end_time = time.time()
    dif_time = end_time - start_time  #計算總花費時間
    print(dif_time)

    # Prepare an image to draw the contours
    contour_image = np.zeros_like(image)  #創建了一個與原始圖像 image 大小相同的全0矩陣,即一個全黑的圖像,並將其賦值給 contour_image 變量，可以在上面單獨繪製輪廓而不影響到原始圖像

    # Draw each contour
    for contour in contours:  #使用 for 循環遍歷所有檢測到的輪廓(contours列表)。
        for x, y in contour:  #對於每個輪廓(contour),它又使用 for 循環遍歷該輪廓上的每個點(x, y)。
            contour_image[x, y] = 255 #對於每個點,它將 contour_image 中對應的像素值設為 255,即白色。

    # Show the resulting image
    cv2.imshow('Processed Image', contour_image)  #顯示結果圖像在processed image中
    cv2.waitKey(0)  #使程序暫停,等待用戶按下任意鍵，可以確保圖像窗口一直保持打開,直到用戶手動關閉它
    cv2.destroyAllWindows()  #在用戶按下任意鍵後,會關閉所有由 OpenCV 創建的窗口


# Replace 'path_to_image.tif' with your image file path
process_image('Test_images/Slight under focus/0066.tiff', 'Test_images/Slight under focus/background.tiff')

# Set the directory containing your files
# directory = 'Test_images/Slight under focus'
# # Get a list of all tiff files
# files = [f for f in os.listdir(directory) if f.endswith('.tiff')]
# for image in files:
#     image_path = os.path.join(directory, image)
#     print(image_path)
#     process_image(image_path, 'Test_images/Slight under focus/background.tiff') #for more image


運作時遇到問題

1. opencv 安裝不了: 
    
    程式碼pip install opencv-python，response: pip is not a function
    
    改善: ▪️python環境變數設定改變  ▪️install a latest version of python 
    

2.result:
模糊後的圖像blurred和模糊背景blurred_bg的維度一致(200, 992)
耗時:325ms



1. **改善後code (加速)**  (version2.py)

```python
import cv2
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor

def trace_contours(edge_image):
    # 使用 OpenCV 的 findContours 函數
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 將輪廓轉換為所需的格式
    return [contour.reshape(-1, 2) for contour in contours]
```

     **改善1** :  trace_contours函數中，以opencv 內建findContours函數替代自訂義函數

原始:                                                                   findcontours:

return 點列表(每個點為x,y表示)                       return numpy數組(每組代表1輪廓)

detect all pixel that’s not 0                                only detect external contours(cv2.RETR_EXTERNA)檢測並返回所有非零像素點作為輪廓             壓縮取回的Contour像素點(不耗內存，快速) python實現                                                            c++實現(編譯執行速度較快)

精準度有差但一般來說大部分應用精準度不會相差太大

缺點是就只能偵測到外部輪廓，如需偵測全部點則使用cv2.RETR_LIST代替cv2.RETR_EXTERNAL(與原程式結果相同

```python
def load_and_blur_image(image_path, kernel_size=(5, 5)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return image, blurred_image
def show_image(name, img):
    cv2.imshow(name, img)   #之後用到之函式定義
def process_image(image_path, background_path):
    # 以灰度模式加載圖形
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    start_time = time.time()  #開始計時
```

**改善2** : 與原程式碼大同小異新版錦江加載與模糊處理放置於同一函數**load_and_blur_image中**

```python
with ThreadPoolExecutor() as executor:
        executor.submit(show_image, 'raw', image)
        
        # Apply Gaussian blur to smooth the image
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        executor.submit(show_image, 'blurred', blurred)

        # Background subtraction
        bg_sub = cv2.subtract(blurred_bg, blurred)
        executor.submit(show_image, 'bg_sub', bg_sub)

        # Apply threshold
        _, binary = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)
        executor.submit(show_image, 'binary', binary)

        # Erode and dilate to remove noise
        dilate1 = cv2.dilate(binary, kernel, iterations=2)
        executor.submit(show_image, 'dilate1', dilate1)

        erode1 = cv2.erode(dilate1, kernel, iterations=2)
        executor.submit(show_image, 'erode1', erode1)

        erode2 = cv2.erode(erode1, kernel, iterations=1)
        executor.submit(show_image, 'erode2', erode2)

        dilate2 = cv2.dilate(erode2, kernel, iterations=1)
        executor.submit(show_image, 'dilate2', dilate2)

        # Apply Canny edge detector to find edges
        edges = cv2.Canny(dilate2, 50, 150)
        executor.submit(show_image, 'canny edges', edges)
        # Trace contours from the edge image
        contours = trace_contours(edges)

        end_time = time.time()
        dif_time = end_time - start_time
        print(dif_time)  #結束計時
```

**改善3** :使用**ThreadPoolExecutor，減少處理時間(並行執行，縮短處理時間)**

時間縮短至約40ms

⚠️problom: 除了raw與最終結果外，其餘圖片跑出後即關閉(閃現)

```python
 # Draw each contour
        cv2.drawContours(contour_image, contours, -1, (255), 1)

    # Show the resulting image
        cv2.imshow('Processed Image', contour_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Replace 'path_to_image.tif' with your image file path
process_image('Test_images/Slight under focus/0066.tiff', 'Test_images/Slight under focus/background.tiff')

# Set the directory containing your files
# directory = 'Test_images/Slight under focus'
# # Get a list of all tiff files
# files = [f for f in os.listdir(directory) if f.endswith('.tiff')]
# for image in files:
#     image_path = os.path.join(directory, image)
#     print(image_path)
#     process_image(image_path, 'Test_images/Slight under focus/background.tiff')
```



***update: 最終程式碼** (latest(version3) correct code.py)

```python
import cv2
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor

def trace_contours(edge_image):
    contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return [contour.reshape(-1, 2) for contour in contours]

def load_and_blur_image(image_path, kernel_size=(5, 5)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return image, blurred_image

def show_image(name, img, images_dict):
    images_dict[name] = img

def process_image(image_path, background_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    images_dict = {}
    show_image('raw', image, images_dict)
    
    start_time = time.time()
    
    with ThreadPoolExecutor() as executor:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        executor.submit(show_image, 'blurred', blurred, images_dict)

        bg_sub = cv2.subtract(blurred_bg, blurred)
        executor.submit(show_image, 'bg_sub', bg_sub, images_dict)

        _, binary = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)
        executor.submit(show_image, 'binary', binary, images_dict)

        dilate1 = cv2.dilate(binary, kernel, iterations=2)
        executor.submit(show_image, 'dilate1', dilate1, images_dict)

        erode1 = cv2.erode(dilate1, kernel, iterations=2)
        executor.submit(show_image, 'erode1', erode1, images_dict)

        erode2 = cv2.erode(erode1, kernel, iterations=1)
        executor.submit(show_image, 'erode2', erode2, images_dict)

        dilate2 = cv2.dilate(erode2, kernel, iterations=1)
        executor.submit(show_image, 'dilate2', dilate2, images_dict)

        edges = cv2.Canny(dilate2, 50, 150)
        executor.submit(show_image, 'canny edges', edges, images_dict)

        contours = trace_contours(edges)

    end_time = time.time()
    dif_time = end_time - start_time
    print(dif_time)

    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (255), 1)
    show_image('Processed Image', contour_image, images_dict)

    while True:
        for name, img in images_dict.items():
            cv2.imshow(name, img)
        
        key = cv2.waitKey(1)
        if key!=-1:          #按下任意鍵
            break

    cv2.destroyAllWindows()

# 使用方法
process_image('Test_images/Slight under focus/0066.tiff', 'Test_images/Slight under focus/background.tiff')

#多個影像方法
# Set the directory containing your files
#directory = 'Test_images/Slight under focus'
# # Get a list of all tiff files
#files = [f for f in os.listdir(directory) if f.endswith('.tiff')]
#for image in files:
#    image_path = os.path.join(directory, image)
#    print(f"Processing: {image_path}")
#    process_image(image_path, 'Test_images/Slight under focus/background.tiff')
#    print("Press any key to continue to the next image...")
```

改善處:

前版本閃現原因可能為無足夠時間使窗口開啟與更新。

因此新版本新增一個while迴圈，且設定每次執更新間先等1ms，這樣才有足夠時間反應，避免未響應而直接被關閉(閃現問題)

至於show image函式式中images_dict[name] = img，images_dict就像一字典，而name是他的要搜尋的單字，而此函數定義內將每一個name都賦予一個相對應的圖。這個過程在上述的while迴圈中，**`images_dict.items()`** 會回傳一個包含字典所有值對的列表。每次更新時，**`name`** 會被賦予字典的鍵(圖名)，**`img`** 會被賦予對應的值（圖像數據）。

如此一來迴圈中的下一步就是將這兩個對應值，利用opencv內鍵函數顯示圖片。

迴圈下方新增一條件句，用意是當用戶按下任意鍵時，關閉窗口且跳到下一個while迴圈執行下一個圖片檔

**result:**

成功將時間縮短至8.34ms，且解決閃現問題。(單圖測試)

多圖片測試部分結果:平均約 10ms