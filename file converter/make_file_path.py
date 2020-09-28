import os

file_path = 'A:/Users/SSY/Desktop/dataset/cud_calibration/200925 dataset/train/A'

path_list = os.listdir(file_path)

with open('make_file_path.txt', 'wb') as f:
    for path in path_list:
        f.write(path.encode())  # 엔터키 안됨
        f.write(b'\n')
