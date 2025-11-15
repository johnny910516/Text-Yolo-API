import os
from tqdm import tqdm
import requests
import shutil
import numpy as np
import base64

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    txt_content = []

    for filenames in os.listdir('./output'):
        file_path = os.path.join('./output', filenames)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path) 
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path) 

    image_list = []
    for image_index, image_path in enumerate(tqdm(image_paths, desc=f'開始檢測 {image_amount} 張影像')):
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        split_data = {
            'image_base64': image_base64,
            'image_path': image_path,
            'image_amount': image_amount,
            'image_index': image_index,
            'filename': filename,
            'device': str(DEVICE), 
        }

        split_response = requests.post("http://localhost:5001/split", json=split_data)

        if split_response.json()['error_code'] == 0:
            print(f'{filename} {split_response.json()['message']}, error_code: {split_response.json()['error_code']}')

            recognize_data = {
                'image_base64_list': split_response.json()['result'],
                'image_amount': image_amount,
                'image_index': image_index,
                'filename': filename,
                'file_ext': file_ext,
                'txt_content': txt_content,
                'device': str(DEVICE)
            }

            recognize_responese = requests.post("http://localhost:5002/recognize", json=recognize_data)

            if recognize_responese.json()['error_code'] == 0:
                print(f'{filename} {recognize_responese.json()['message']}, error_code: {recognize_responese.json()['error_code']}')
                txt_content = recognize_responese.json()['result']['txt_content']
                part1 = recognize_responese.json()['result']['part1']
                part2 = recognize_responese.json()['result']['part2']
            else:
                print(f'{filename} {recognize_responese.json()['message']}, error_code: {recognize_responese.json()['error_code']}')

        else:
            print(f'{filename} {split_response.json()['message']}, error_code: {split_response.json()['error_code']}')

