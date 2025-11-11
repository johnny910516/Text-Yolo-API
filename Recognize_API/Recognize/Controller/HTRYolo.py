import cv2
import os
import numpy as np
from PIL import Image
import argparse
from collections import OrderedDict
from pathlib import Path
import yaml
from ultralytics import YOLO

class HTRYolo():
    def split_txt_by_threshold(self, root, txt_filename ,filename, threshold):
        input_file = os.path.join(root, txt_filename)

        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        parts = text.split("##")

        cumulative_length = 0
        split_index = None

        for i in range(len(parts) - 1):
            cumulative_length += len(parts[i])
            if cumulative_length >= threshold:
                split_index = i
                break

        output1 = os.path.join(root, (f"{filename}_part1.txt"))
        output2 = os.path.join(root, (f"{filename}_part2.txt"))

        if split_index is None:
            part1 = text.strip() + "**" 
            part2 = ""
        else:
            part1 = "##".join(parts[:split_index + 1]).strip() + "**"
            part2 = "##".join(parts[split_index + 1:]).lstrip("\n")

        with open(output1, "w", encoding="utf-8") as f:
            f.write(part1)
        with open(output2, "w", encoding="utf-8") as f:
            f.write(part2)

    def write_txt(self, text, output_path, filename):
        print(os.path.join(output_path, f'{filename}.txt'))
        with open(os.path.join(output_path, f'{filename}.txt'), 'w', encoding='utf-8') as file:
            file.write(text)

    def htr(self, args, htr_model, htr_image_path, txt_content, device):
        if len(htr_image_path) == 1:
            txt_content.append(htr_image_path)
            return txt_content

        htr_image = cv2.imread(htr_image_path)

        try:
            predictions = htr_model.predict(
                source=htr_image,
                device=device,
                verbose=False,
                imgsz=args.htr_size,
                save=False
            )

            recognized_text = ''
            for i, prediction in enumerate(predictions):
                top1_class_id = int(prediction.probs.top1)
                recognized_text += prediction.names.get(top1_class_id, "")

        except Exception as e:
            print(f"HTR 識別過程中發生錯誤: {e}")
            recognized_text = ''

        # 將結果加進 txt_content[0]
        txt_content.append(recognized_text)
        return txt_content

    def split_postprocesser(self, image_list, post_process_split_path):
        index = 0
        post_process_image_list = []

        for pil_image in image_list:
            if isinstance(pil_image, str):
                post_process_image_list.append(pil_image)
                continue
            image = np.array(pil_image)
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            h, w = binary_img.shape
            target_size = 64

            if h > target_size or w > target_size:
                scale = min(target_size / w, target_size / h)
                new_w, new_h = int(w * scale), int(h * scale)
            else:  
                new_w, new_h = w, h

            if new_w == 0: new_w = 1
            if new_h == 0: new_h = 1

            if new_w != w or new_h != h:
                resized_binary = cv2.resize(binary_img, (new_w, new_h),
                                            interpolation=cv2.INTER_AREA if (
                                                    new_w < w or new_h < h) else cv2.INTER_CUBIC)
            else:
                resized_binary = binary_img

            final_binary = np.ones((target_size, target_size), dtype=np.uint8) * 255
            h_r, w_r = resized_binary.shape
            y_offset = (target_size - h_r) // 2
            x_offset = (target_size - w_r) // 2
            final_binary[y_offset:y_offset + h_r, x_offset:x_offset + w_r] = resized_binary

            htr_input_img = cv2.cvtColor(final_binary, cv2.COLOR_GRAY2BGR)

            name = f"{str(index).zfill(4)}"

            cv2.imwrite(f"{post_process_split_path}/{name}.jpg", htr_input_img)
            index += 1

            post_process_image_list.append(Image.fromarray(htr_input_img))

        return post_process_image_list

    def check_split_word_resolution(self, image_paths):
        resolution_list = []
        for file in os.listdir(image_paths):
            image = Image.open(os.path.join(image_paths, file))
            resolution_list.append(max(image.width, image.height))
        
        return sum(resolution_list) / len(resolution_list)

    def split(self, sort_text_coordinate, image, split_file_path):
        index=0
        image_list = []
        for coordinate in sort_text_coordinate:
            if isinstance(coordinate, str):
                image_list.append(coordinate)
                continue
            if (coordinate < 0).any():
                continue
            img = image[coordinate[0][1]:coordinate[2][1], coordinate[0][0]:coordinate[2][0]]
            name = f"{str(index).zfill(4)}"
            cv2.imwrite(f"{split_file_path}/{name}.jpg", img)
            index += 1

            image_list.append(Image.fromarray(img))

        return image_list
    
    def load_model(self, args, project_root):
        print('Load model...')

        htr_model = YOLO(project_root/args.htr_weight)

        return htr_model
    
    def load_papper_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return data

    def get_argparse(self):
        parser = argparse.ArgumentParser(description='Text Detection')
        #稿紙參數檔路徑
        parser.add_argument('--papper_config', default=r'./config/papper_config.yaml', type=str, help='稿紙參數檔路徑')

        #模型權重路徑
        parser.add_argument('--htr_weight', default=r'./weight/htr.pt', type=str, help='手寫字辨識模型權重')

        #測試模式閾值
        parser.add_argument('--resoultion_threshold', default=20, type=int, help='切割文字解析度閾值(若test_mode為True則會將切割文字平均解析度低於閾值的影像過濾不做)') 

        #模型參數
        parser.add_argument("--htr_size", type=int, default=64, help="手寫字辨識影像縮放尺寸")

        #影像輸入/輸出路徑
        parser.add_argument('--output', default=r'./output', type=str, help='預測輸出路徑') 
        parser.add_argument('--split_output', default='split', type=str, help='切割文字輸出路徑') 
        parser.add_argument('--post_procss_split_output', default='post process split', type=str, help='後處理切割文字輸出路徑') 

        return parser.parse_args()
    
    def predict(self, image_amount, image_index, filename, file_ext, post_process_image_path_list, txt_content, device):
        current_dir = Path(__file__).resolve().parent 
        project_root = current_dir.parents[2]  
        
        args = self.get_argparse()
        data = self.load_papper_config(project_root/args.papper_config)

        args.column = data['column']
        args.row = data['row']
        args.example_format = data['example format']
        args.high_school_format = data['high school format']
        args.test_mode = data['test mode']

        htr_model = self.load_model(args, project_root)

        split_path = os.path.join(project_root/args.output, filename, args.split_output)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        
        print(split_path)

        if args.test_mode:
            average_resolution = self.check_split_word_resolution(split_path)
            print(average_resolution)
            if average_resolution > args.resoultion_threshold:
                for htr_image_path in post_process_image_path_list:
                    txt_content = self.htr(args, htr_model, htr_image_path, txt_content, device)
                if image_index == image_amount-1:
                    print(image_index)
                    self.write_txt(''.join(txt_content), str(project_root/args.output), 'recongnize')
                
                if args.high_school_format:
                    self.split_txt_by_threshold(str(project_root/args.output), 'recongnize.txt', filename, args.column)
            else:
                print(f"-----{filename}{file_ext}-----")
                print(f'平均解析度: {average_resolution}')
                print("解析度不足!!!")
                print("請重新拍攝!!!")
                print()
        else:
            for htr_image_path in post_process_image_path_list:
                txt_content = self.htr(args, htr_model, htr_image_path, txt_content, device)
            if image_index == image_amount-1:
                self.write_txt(''.join(txt_content), str(project_root/args.output), 'recongnize')

            if args.high_school_format:
                self.split_txt_by_threshold(str(project_root/args.output), 'recongnize.txt', filename, args.column)
        
        response = OrderedDict({
            'success': True,
            'txt_content': txt_content
        })

        return response