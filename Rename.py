import os

#自動遍歷指定目錄下的所有 .tiff 格式的文件,並將它們重命名為按序編號的文件名

# Set the directory containing your files
directory = 'Test_images/Slight under focus'  #directory 變量,用於指定要處理的目錄路徑，Test_images/Slight under focusj為目錄名

# Get a list of all tiff files
files = [f for f in os.listdir(directory) if f.endswith('.tiff')]  #使用 os.listdir() 函數獲取目錄中所有文件的列表,並遍歷該列表,僅選擇文件名以 .tiff 結尾的文件。這些文件被存儲在 files 變量中
files.sort()  # Sort the files to maintain any existing order, if necessary 對 files 列表進行了排序

# Rename files sequentially
for index, filename in enumerate(files):  #同時獲得每個文件的索引(index)和文件名(filename)。
    new_filename = f"{index:04d}.tiff"  # Generates '0000.tiff', '0001.tiff', etc.  #們使用 f-string 格式化的方式創建新的文件名,如 "0000.tiff"、"0001.tiff" 等。
    old_file = os.path.join(directory, filename)  #os.path.join(directory, filename) 用於拼接目錄路徑和文件名,獲得完整的舊文件路徑(old_file)。
    new_file = os.path.join(directory, new_filename)  #os.path.join(directory, new_filename) 用於構建新文件路徑(new_file)

    os.rename(old_file, new_file)  #os.rename(old_file, new_file) 函數用於將舊文件重命名為新文件名
    print(f"Renamed '{filename}' to '{new_filename}'")

print("Renaming complete.")