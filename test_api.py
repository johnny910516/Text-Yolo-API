import os
from tqdm import tqdm
import requests
import shutil
import numpy as np

def load_image_path(folder_path):
        image_paths = []
        for image_name in os.listdir(folder_path):
            if image_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(folder_path, image_name)
                image_paths.append(image_path)

        return image_paths

if __name__ == '__main__':
    image_paths = load_image_path('./input')
    image_amount = len(image_paths)

    for filenames in os.listdir('./output'):
        file_path = os.path.join('./output', filenames)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path) 
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path) 

    for image_index, image_path in enumerate(tqdm(image_paths, desc=f'開始檢測 {image_amount} 張影像')):
        filename, file_ext = os.path.splitext(os.path.basename(image_path))

        data = {
             'image_path': image_path,
             'image_index': image_index,
             'image_amount': image_amount,
             'filename': filename,
        }

        response = requests.post("http://localhost:5001/post", data=data)
        print(f'\n{filename}{file_ext} {response.json()['message']}, error code: {response.json()['error_code']}')
            
