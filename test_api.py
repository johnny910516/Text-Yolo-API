import os
from tqdm import tqdm
import requests
import shutil
import numpy as np
import base64

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_image_path(folder_path):
    image_path_list = []
    image_base64_list = []
    for image_name in os.listdir(folder_path):
        if image_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, image_name)
            image_path_list.append(image_path)

            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
                image_base64_list.append(image_base64)

    return image_path_list, image_base64_list

if __name__ == '__main__':
    for filenames in os.listdir('./output'):
        file_path = os.path.join('./output', filenames)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path) 
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path) 

    txt_content = []
    image_path_list, image_base64_list = load_image_path('./input')
    image_amount = len(image_path_list)

    split_data = {
        'image_base64_list': image_base64_list,
        'image_path_list': image_path_list,
        'image_amount': image_amount,
        'device': str(DEVICE)
    }

    split_response = requests.post("http://localhost:5001/split", json=split_data)

    if split_response.json()['error_code'] == 0:
        print(f'{split_response.json()['message']}, error_code: {split_response.json()['error_code']}')

        recognize_data = {
            'image_base64_lists': split_response.json()['result'],
            'image_path_list': image_path_list,
            'image_amount': image_amount,
            'device': str(DEVICE)
        }

        recognize_responese = requests.post("http://localhost:5002/recognize", json=recognize_data)

        if recognize_responese.json()['error_code'] == 0:
            print(f'{recognize_responese.json()['message']}, error_code: {recognize_responese.json()['error_code']}')
            txt_content = ''.join(recognize_responese.json()['result']['txt_content'])
            part1 = recognize_responese.json()['result']['part1']
            part2 = recognize_responese.json()['result']['part2']
        else:
            print(f'{recognize_responese.json()['message']}, error_code: {recognize_responese.json()['error_code']}')

    else:
        print(f'{split_response.json()['message']}, error_code: {split_response.json()['error_code']}')

