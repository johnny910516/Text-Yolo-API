import cv2
import numpy as np
from PIL import Image
import os
from PIL import Image
import argparse
from scipy.spatial import distance
import os
import cv2
import numpy as np
from collections import OrderedDict
from ultralytics import YOLO
import yaml
from pathlib import Path
import math
import base64
import re

class TextYolo():    
    def split_txt_by_threshold(self, input_file, threshold):
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

        output1 = input_file.replace(".txt", "_part1.txt")
        output2 = input_file.replace(".txt", "_part2.txt")

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

        with open(os.path.join(output_path, f'{filename}.txt'), 'w', encoding='utf-8') as file:
            file.write(text)
    
    def htr(self, args, htr_model, htr_image, txt_content, device):
        if isinstance(htr_image, str):
            txt_content[0] += htr_image
            return txt_content
        
        try:
            predictions = htr_model.predict(
                source=htr_image,
                device=device,
                verbose=False,
                imgsz=args.htr_size,
                save=False
            )
            
            for i, prediction in enumerate(predictions):
                top1_class_id = int(prediction.probs.top1)
                recognized_text = prediction.names.get(top1_class_id, "")
                    
        except Exception as e:
            print(f"HTR 識別過程中發生錯誤: {e}")
            recognized_text = ''
        
        txt_content[0] += recognized_text
        return txt_content

    def split_postprocesser(self, args, image_list, post_process_split_path):
        index = 0
        image_base64_list = []

        for pil_image in image_list:
            if isinstance(pil_image, str):
                image_base64_list.append(pil_image)
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
            
            if args.debug_mode:
                name = f"{str(index).zfill(4)}"
                cv2.imwrite(f"{post_process_split_path}/{name}.jpg", htr_input_img)
                index += 1

            _, buffer = cv2.imencode('.png', htr_input_img)  
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            image_base64_list.append(image_base64)

        return image_base64_list
    
    def process_image(self, image, center_size=48):
        """檢查圖片中心區域的非黑像素比例"""
        print(type(image))
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        height, width = binary.shape
        half = center_size // 2
        center_region = binary[
                        max(0, height // 2 - half):min(height, height // 2 + half),
                        max(0, width // 2 - half):min(width, width // 2 + half)
                        ]
        total_pixels = center_region.size
        non_black_pixels = cv2.countNonZero(center_region)
        percentage = (non_black_pixels / total_pixels) * 100

        return percentage
    
    def detect_img_white(self, image, coordinate):
        xs = [pt[0] for pt in coordinate]
        ys = [pt[1] for pt in coordinate]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # NumPy 影像 slicing：image[y, x]
        crop = image[y_min:y_max, x_min:x_max]

        percentage = self.process_image(crop)

        return coordinate if percentage < 99 else None

    def split(self, args, sort_text_coordinate, image, split_file_path):
        index = 0
        image_list = []

        for coordinate in sort_text_coordinate:
            if isinstance(coordinate, str):
                image_list.append(coordinate)
                continue

            if any(pt[0] < 0 or pt[1] < 0 for pt in coordinate):
                continue

            x_values = [pt[0] for pt in coordinate]
            y_values = [pt[1] for pt in coordinate]
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)

            if x_min == x_max or y_min == y_max:
                continue

            img_crop = [row[x_min:x_max] for row in image[y_min:y_max]]
            img_crop = Image.fromarray(np.array(img_crop)) 

            if args.debug_mode:
                name = f"{str(index).zfill(4)}.jpg"
                img_crop.save(f"{split_file_path}/{name}")

            index += 1
            image_list.append(img_crop)

        return image_list
    
    def serach_insert_coordinate(self, image, caret_coordinate, caret_direction):
        img = image[caret_coordinate[0][1]:caret_coordinate[2][1], caret_coordinate[0][0]:caret_coordinate[1][0]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img

        blur = cv2.GaussianBlur(gray, (5, 5), 0)  

        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        y_insert_position, x_insert_position = np.where(binary == 0)
        if len(x_insert_position) > 0:
            if caret_direction == 'right':
                x = np.min(x_insert_position)
                x_indices = np.where(x_insert_position == x)[0]
            elif caret_direction == 'left':
                x = np.max(x_insert_position)
                x_indices = np.where(x_insert_position == x)[0]

            y_avg = int(np.mean(y_insert_position[x_indices]))
            
            x_insert_position = x + caret_coordinate[0][0]
            y_insert_position = y_avg + caret_coordinate[0][1]

            cv2.circle(image, (x_insert_position, y_insert_position), radius=2, color=(0,0,255), thickness=2)
    
        return x_insert_position, y_insert_position

    def search_caret_coordinate_insert_position(self, args, insert_caret_point_image, new_sort_text_coordinate, caret_dict, insert_caret_point_path):
        def caret_mark_direction_detection(coordinate_list):
            # 判斷 caret 的方向
            if coordinate_list[1] is not None:
                big_left = min([pt[0] for pt in coordinate_list[0]])
                big_right = max([pt[0] for pt in coordinate_list[0]])
                small_center_x = sum([pt[0] for pt in coordinate_list[1]]) / len(coordinate_list[1])

                dist_to_left = abs(small_center_x - big_left)
                dist_to_right = abs(small_center_x - big_right)

                return "right" if dist_to_left < dist_to_right else "left"
            else:
                if len(coordinate_list) > 2:
                    big_left = min([pt[0] for pt in coordinate_list[0]])
                    big_right = max([pt[0] for pt in coordinate_list[0]])
                    small_center_x = sum([pt[0] for pt in coordinate_list[2]]) / len(coordinate_list[2])

                    dist_to_left = abs(small_center_x - big_left)
                    dist_to_right = abs(small_center_x - big_right)

                    return "right" if dist_to_left < dist_to_right else "left"

                return None

        for key, value in caret_dict.items():
            caret_text_coordinates = [value[i] for i in range(2, len(value))]
            caret_mark_direction = caret_mark_direction_detection(value)

            if caret_mark_direction == 'right':
                caret_text_coordinates.sort(key=lambda quad: quad[0][1])
                if value[1] is not None:
                    x_insert_position, y_insert_position = self.serach_insert_coordinate(insert_caret_point_image, value[1], 'right')
                else:
                    x_insert_position = value[2][3][0]
                    y_insert_position = value[2][3][1]

                min_dist = float('inf')
                insert_index = None
                insert_poly = None

                for i, poly in enumerate(new_sort_text_coordinate):
                    if isinstance(poly, str):
                        continue

                    br_x = max(pt[0] for pt in poly)
                    br_y = max(pt[1] for pt in poly)
                    dist = ((x_insert_position - br_x) ** 2 + (y_insert_position - br_y) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        insert_poly = poly
                        insert_index = i

                if y_insert_position < insert_poly[0][1]:
                    insert_index -= 1

                for caret_text_coordinate in caret_text_coordinates:
                    insert_index += 1
                    new_sort_text_coordinate.insert(insert_index, caret_text_coordinate)

            elif caret_mark_direction == 'left':
                caret_text_coordinates.sort(key=lambda quad: quad[0][1])
                if value[1] is not None:
                    x_insert_position, y_insert_position = self.serach_insert_coordinate(insert_caret_point_image, value[1], 'left')
                else:
                    x_insert_position = value[2][2][0]
                    y_insert_position = value[2][2][1]

                min_dist = float('inf')
                insert_index = None
                insert_poly = None

                for i, poly in enumerate(new_sort_text_coordinate):
                    if isinstance(poly, str):
                        continue

                    bl_x = min(pt[0] for pt in poly)
                    bl_y = max(pt[1] for pt in poly)
                    dist = ((x_insert_position - bl_x) ** 2 + (y_insert_position - bl_y) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        insert_poly = poly
                        insert_index = i

                if y_insert_position < insert_poly[0][1]:
                    insert_index -= 1

                for caret_text_coordinate in caret_text_coordinates:
                    insert_index += 1
                    new_sort_text_coordinate.insert(insert_index, caret_text_coordinate)

        if args.debug_mode:
            cv2.imwrite(insert_caret_point_path, insert_caret_point_image)

        return new_sort_text_coordinate
    
    def poly_to_bbox(self, poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
    
        return [min(xs), min(ys), max(xs), max(ys)]
    
    def iou(self, poly1, poly2):
        box1 = self.poly_to_bbox(poly1)
        box2 = self.poly_to_bbox(poly2)
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0
    
    def match_iou_max_no_threshold(self, A, B):
        unmatched_A_idx = set(range(len(A)))
        unmatched_B_idx = set(range(len(B)))
        matches = []

        # 建立 IoU 矩陣 (list of lists)
        iou_matrix = [[self.iou(a, b) for b in B] for a in A]

        # 貪婪匹配
        while True:
            max_iou = -1
            max_i = None
            max_j = None

            # 找出最大 IoU 及索引
            for i, row in enumerate(iou_matrix):
                for j, val in enumerate(row):
                    if val > max_iou:
                        max_iou = val
                        max_i = i
                        max_j = j

            if max_i is None or max_j is None or max_iou == -1:
                break

            matches.append((max_i, max_j))
            unmatched_A_idx.discard(max_i)
            unmatched_B_idx.discard(max_j)

            # 標記為已匹配
            for j in range(len(B)):
                iou_matrix[max_i][j] = -1
            for i in range(len(A)):
                iou_matrix[i][max_j] = -1

        return list(unmatched_A_idx), list(unmatched_B_idx)
    
    def high_school_insert_paragraph_mark(self, args, image_index, image_amount, sort_text_coordinate, sort_textbox_coordinate):
        new_sort_text_coordinate = []
        text_textbox_coordinate = []
        post_unmatched_A_idx = []
        empty_column_amount = 0

        for i in range(args.row):
            try:
                textbox_list = sort_textbox_coordinate[args.column*i : args.column*(i+1)]
                text_list = sort_text_coordinate[:args.column+5]

                all_points = [pt for box in textbox_list for pt in box]
                xs = [pt[0] for pt in all_points]
                ys = [pt[1] for pt in all_points]
                min_x, min_y = min(xs), min(ys)
                max_x, max_y = max(xs), max(ys)

                points_in_range = []
                for text in text_list:
                    xs_text = [pt[0] for pt in text]
                    if all(min_x-10 <= x <= max_x+10 for x in xs_text):
                        points_in_range.append(text)

                sort_text_coordinate = [a for a in sort_text_coordinate if a not in points_in_range]

                unmatched_A_idx, unmatched_B_idx = self.match_iou_max_no_threshold(textbox_list, points_in_range)

                tmp_unmatched_A_idx = unmatched_A_idx.copy()
                if 0 in tmp_unmatched_A_idx and 1 in tmp_unmatched_A_idx:
                    tmp_unmatched_A_idx = [x for x in tmp_unmatched_A_idx if x not in (0, 1)]

                # 如果整個直排空白
                if set(unmatched_A_idx) == set(range(args.column)) and len(sort_text_coordinate) == 0:
                    new_sort_text_coordinate.extend(['*', '*'])
                    break
                elif set(unmatched_A_idx) == set(range(args.column)) and len(sort_text_coordinate) != 0:
                    empty_column_amount += 1
                    if empty_column_amount == 1 and len(new_sort_text_coordinate) > 0 and not isinstance(new_sort_text_coordinate[-1], str):
                        new_sort_text_coordinate.extend(['#', '#'])
                        new_sort_text_coordinate.append('\n')
                    continue

                # 段落開頭
                if set(unmatched_A_idx) == {0, 1}:
                    if (len(post_unmatched_A_idx) == 0 and not (image_index == 0 and i == 0)) or set(post_unmatched_A_idx) == {0, 1}:
                        new_sort_text_coordinate.extend(['#', '#'])
                        new_sort_text_coordinate.append('\n')
                    new_sort_text_coordinate.extend([' ', ' ', ' ', ' '])
                    text_textbox_coordinate.extend([textbox_list[0], textbox_list[0]])
                elif len(tmp_unmatched_A_idx) != 0 and set(tmp_unmatched_A_idx) == set(range(tmp_unmatched_A_idx[0], args.column)):
                    if {0, 1}.issubset(unmatched_A_idx):
                        new_sort_text_coordinate.extend([' ', ' ', ' ', ' '])
                        text_textbox_coordinate.extend([textbox_list[0], textbox_list[0]])
                    new_sort_text_coordinate.extend(points_in_range)
                    text_textbox_coordinate.extend(points_in_range)
                    if len(sort_text_coordinate) != 0:
                        new_sort_text_coordinate.extend(['#', '#'])
                        new_sort_text_coordinate.append('\n')

                    post_unmatched_A_idx = unmatched_A_idx.copy()
                    continue
                # 缺字處理
                elif len(unmatched_A_idx) != 0 and set(unmatched_A_idx) != set(range(unmatched_A_idx[0], args.column)):
                    points_in_range_index = 0
                    flag1 = False
                    flag2 = False
                    for insert_index in range(args.column):
                        if insert_index in unmatched_A_idx:
                            if {0, 1}.issubset(unmatched_A_idx) and not flag2:
                                if (len(post_unmatched_A_idx) == 0 and not (image_index == 0 and i == 0)) or set(post_unmatched_A_idx) == {0, 1}:
                                    new_sort_text_coordinate.extend(['#', '#'])
                                    new_sort_text_coordinate.append('\n')
                                    flag2 = True
                            if not flag1:
                                new_sort_text_coordinate.extend([' ', ' '])
                                text_textbox_coordinate.append(textbox_list[insert_index])
                        else:
                            new_sort_text_coordinate.append(points_in_range[points_in_range_index])
                            text_textbox_coordinate.append(points_in_range[points_in_range_index])
                            points_in_range_index += 1
                            if len(points_in_range) == points_in_range_index:
                                flag1 = True

                    if (args.column-1) in unmatched_A_idx:
                        new_sort_text_coordinate.extend(['#', '#'])
                        new_sort_text_coordinate.append('\n')
                    post_unmatched_A_idx = unmatched_A_idx.copy()
                    continue

                new_sort_text_coordinate.extend(points_in_range)
                text_textbox_coordinate.extend(points_in_range)
                post_unmatched_A_idx = unmatched_A_idx.copy()

            except Exception as e:
                print(f"[Error] 第 {i} 行發生錯誤: {e}")
                continue

        # 如果整張影像寫滿
        if image_index == (image_amount-1) and i == args.row-1:
            if not isinstance(new_sort_text_coordinate[-1], str) and not isinstance(new_sort_text_coordinate[-2], str):
                new_sort_text_coordinate.extend(['*', '*'])

        return new_sort_text_coordinate, text_textbox_coordinate

    def insert_paragraph_mark(self, args, image_index, image_amount, sort_text_coordinate, sort_textbox_coordinate):
        new_sort_text_coordinate = []
        text_textbox_coordinate = []
        post_textbox_list = []
        post_points_in_range= []
        post_unmatched_A_idx = []

        for i in range(args.row):
            try:
                textbox_list = sort_textbox_coordinate[args.column*i : args.column*(i+1)]
                text_list = sort_text_coordinate[:args.column+5]

                all_points = [pt for box in textbox_list for pt in box]
                xs = [pt[0] for pt in all_points]
                ys = [pt[1] for pt in all_points]
                min_x, min_y = min(xs), min(ys)
                max_x, max_y = max(xs), max(ys)

                points_in_range = []
                for text in text_list:
                    xs_text = [pt[0] for pt in text]
                    if all(min_x-10 <= x <= max_x+10 for x in xs_text):
                        points_in_range.append(text)

                sort_text_coordinate = [a for a in sort_text_coordinate if a not in points_in_range]

                unmatched_A_idx, unmatched_B_idx = self.match_iou_max_no_threshold(textbox_list, points_in_range)

                # 如果整個直排空白
                if set(unmatched_A_idx) == set(range(args.column)) and len(sort_text_coordinate) == 0:
                    new_sort_text_coordinate.extend(['*', '*'])
                    break

                # 第一張影像的第一直排先跳過
                if image_index == 0 and i == 0:
                    pass

                # 用第一張影像的第二直排判斷第一直是否為標題
                elif image_index == 0 and i == 1:
                    if set(unmatched_A_idx) == {0, 1}:
                        new_sort_text_coordinate.extend(post_points_in_range)
                        new_sort_text_coordinate.extend(['@', '@', '\n', ' ', ' ', ' ', ' '])
                        text_textbox_coordinate.extend(post_points_in_range)
                        text_textbox_coordinate.extend([textbox_list[0], textbox_list[1]])

                    elif {0, 1}.issubset(unmatched_A_idx):
                        new_sort_text_coordinate.extend(post_points_in_range)
                        new_sort_text_coordinate.extend(['@', '@', '\n'])
                        text_textbox_coordinate.extend(post_points_in_range)
                        points_in_range_index = 0
                        flag1 = False
                        for insert_index in range(args.column):
                            if insert_index in unmatched_A_idx:
                                if not flag1:
                                    new_sort_text_coordinate.extend([' ', ' '])
                                    text_textbox_coordinate.append(textbox_list[insert_index])
                            else:
                                new_sort_text_coordinate.append(points_in_range[points_in_range_index])
                                text_textbox_coordinate.append(points_in_range[points_in_range_index])
                                points_in_range_index += 1
                                if len(points_in_range) == points_in_range_index:
                                    flag1 = True
                        continue

                    else:
                        if len(post_unmatched_A_idx) != 0 and set(post_unmatched_A_idx) != set(range(post_unmatched_A_idx[0], args.column)):
                            points_in_range_index = 0
                            flag1 = False
                            for insert_index in range(args.column):
                                if insert_index in post_unmatched_A_idx:
                                    if not flag1:
                                        new_sort_text_coordinate.extend([' ', ' '])
                                        text_textbox_coordinate.append(post_textbox_list[insert_index])
                                else:
                                    new_sort_text_coordinate.append(post_points_in_range[points_in_range_index])
                                    text_textbox_coordinate.append(post_points_in_range[points_in_range_index])
                                    points_in_range_index += 1
                                    if len(post_points_in_range) == points_in_range_index:
                                        flag1 = True
                        else:
                            for insert_index in range(4):
                                new_sort_text_coordinate.insert(insert_index, ' ')
                            for insert_index in range(2):
                                text_textbox_coordinate.insert(insert_index, post_textbox_list[insert_index])
                            new_sort_text_coordinate.extend(post_points_in_range)
                            text_textbox_coordinate.extend(post_points_in_range)

                        if len(unmatched_A_idx) != 0 and set(unmatched_A_idx) != set(range(unmatched_A_idx[0], args.column)):
                            points_in_range_index = 0
                            flag1 = False
                            for insert_index in range(args.column):
                                if insert_index in unmatched_A_idx:
                                    if not flag1:
                                        new_sort_text_coordinate.extend([' ', ' '])
                                        text_textbox_coordinate.append(textbox_list[insert_index])
                                else:
                                    new_sort_text_coordinate.append(points_in_range[points_in_range_index])
                                    text_textbox_coordinate.append(points_in_range[points_in_range_index])
                                    points_in_range_index += 1
                                    if len(points_in_range) == points_in_range_index:
                                        flag1 = True
                            continue
                        else:
                            text_textbox_coordinate.extend(points_in_range)

                # 段落開頭
                elif set(unmatched_A_idx) == {0, 1}:
                    new_sort_text_coordinate.extend(['#', '#'])
                    new_sort_text_coordinate.append('\n')
                    new_sort_text_coordinate.extend([' ', ' ', ' ', ' '])
                    text_textbox_coordinate.extend([textbox_list[0], textbox_list[0]])

                # 缺字處理
                elif len(unmatched_A_idx) != 0 and set(unmatched_A_idx) != set(range(unmatched_A_idx[0], args.column)):
                    points_in_range_index = 0
                    flag1 = False
                    flag2 = True
                    for insert_index in range(args.column):
                        if insert_index in unmatched_A_idx:
                            if {0, 1}.issubset(unmatched_A_idx) and flag2:
                                new_sort_text_coordinate.extend(['#', '#'])
                                new_sort_text_coordinate.append('\n')
                                flag2 = False
                            if not flag1:
                                new_sort_text_coordinate.extend([' ', ' '])
                                text_textbox_coordinate.append(textbox_list[insert_index])
                        else:
                            new_sort_text_coordinate.append(points_in_range[points_in_range_index])
                            text_textbox_coordinate.append(points_in_range[points_in_range_index])
                            points_in_range_index += 1
                            if len(points_in_range) == points_in_range_index:
                                flag1 = True
                    continue

                if not (image_index == 0 and i == 0):
                    new_sort_text_coordinate.extend(points_in_range)
                    text_textbox_coordinate.extend(points_in_range)

                post_textbox_list.extend(textbox_list)
                post_points_in_range.extend(points_in_range)
                post_unmatched_A_idx.extend(unmatched_A_idx)

            except Exception as e:
                print(f"[Error] 第 {i} 行發生錯誤: {e}")
                continue

        # 如果整張影像寫滿
        if image_index == (image_amount-1) and i == args.row-1:
            if not isinstance(new_sort_text_coordinate[-1], str) and not isinstance(new_sort_text_coordinate[-2], str):
                new_sort_text_coordinate.extend(['*', '*'])

        return new_sort_text_coordinate, text_textbox_coordinate
    
    def sort_text(self, args, sort_textbox_coordinate, text_coordinates):
        sort_text_coordinate = []

        for i in range(args.row):
            # 以每個文字區塊左上角 x 座標排序（反向）
            text_coordinates = sorted(text_coordinates, key=lambda quad: quad[1][0], reverse=True)
            textbox_list = sort_textbox_coordinate[args.column*i:args.column*(i+1)]
            text_list = text_coordinates[:args.column+20]

            # 計算 textbox 的邊界
            all_x = [point[0] for poly in textbox_list for point in poly]
            all_y = [point[1] for poly in textbox_list for point in poly]
            min_x, min_y = min(all_x), min(all_y)
            max_x, max_y = max(all_x), max(all_y)

            # 篩選落在 textbox 範圍內的文字
            points_in_range = []
            for text in text_list:
                xs = [p[0] for p in text]
                if all(min_x-10 <= x <= max_x+10 for x in xs):
                    points_in_range.append(text)

            unmatched_A_idx, unmatched_B_idx = self.match_iou_max_no_threshold(textbox_list, points_in_range)

            # 從 text_coordinates 移除已匹配文字
            text_coordinates = [a for a in text_coordinates if not any(a == b for b in points_in_range)]

            # 移除 unmatched_B_idx 的文字
            points_in_range = [b for idx, b in enumerate(points_in_range) if idx not in unmatched_B_idx]

            # 以左上角 y 座標排序
            points_in_range = sorted(points_in_range, key=lambda quad: quad[0][1])
            sort_text_coordinate.extend(points_in_range)

        return sort_text_coordinate

    def saveResult(self, args, image_index, image_amount, filename, image, sort_textbox_coordinate, text_coordinate, caret_dict, output_path, split_path, text_textbox_split_path, post_process_split_path):
        image = np.array(image)
        draw_text_image = image.copy()
        insert_caret_point_image = image.copy()

        insert_caret_point_path = os.path.join(output_path, f'{filename}_insert_caret_point.jpg')
        text_path = os.path.join(output_path, f'sort_{filename}_text.jpg')

        text_coordinates_list = []
        for coordinate in text_coordinate:
            text_coordinates_list.append([list(pt) for pt in coordinate])

        if args.noise:
            fliter_text_coordinate_list = []
            for coordinate in text_coordinates_list:
                coordinate = self.detect_img_white(image, coordinate)
                if coordinate is not None:
                    fliter_text_coordinate_list.append(coordinate)

            text_coordinates_list = fliter_text_coordinate_list
            
        sort_text_coordinate = self.sort_text(args, sort_textbox_coordinate, text_coordinates_list)

        if args.high_school_format:
            new_sort_text_coordinate, text_textbox_coordinates = self.high_school_insert_paragraph_mark(args, image_index, image_amount, sort_text_coordinate, sort_textbox_coordinate)
        else:
            new_sort_text_coordinate, text_textbox_coordinates = self.insert_paragraph_mark(args, image_index, image_amount, sort_text_coordinate, sort_textbox_coordinate)

        text_textbox_coordinate_list = []
        for coordinate in text_textbox_coordinates:
            text_textbox_coordinate_list.append([list(pt) for pt in coordinate])

        text_coordinate = self.search_caret_coordinate_insert_position(args, insert_caret_point_image, new_sort_text_coordinate, caret_dict, insert_caret_point_path)
        text_textbox_coordinate = self.search_caret_coordinate_insert_position(args, insert_caret_point_image, text_textbox_coordinate_list, caret_dict, insert_caret_point_path)

        split_image = image.copy()
        image_list = self.split(args, text_coordinate, split_image, split_path)
        print(text_textbox_coordinate[0], text_textbox_coordinate[1])
        _ = self.split(args, text_textbox_coordinate, split_image, text_textbox_split_path)
        image_base64_list = self.split_postprocesser(args, image_list, post_process_split_path)

        if args.debug_mode:
            for i, poly in enumerate(text_coordinate, 1):
                if isinstance(poly, str):
                    continue
                poly_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(draw_text_image, [poly_np], True, color=(255, 0, 0), thickness=2)
                if len(poly) > 1:
                    cv2.putText(draw_text_image, str(i), (poly[1][0] - 3, poly[1][1] + 3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)
                else:
                    cv2.putText(draw_text_image, str(i), (poly[0][0], poly[0][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)

            cv2.imwrite(text_path, draw_text_image)

        return image_base64_list

    def filter_coordinate(self, all_text_coordinate, caret_dict, overlap_thresh=0.8):
        caret_text_coordinates = []
        text_coordinates = []

        for text_coordinate in all_text_coordinate:
            text_coordinate_np = np.array(text_coordinate, dtype=np.float32)

            small_area = cv2.contourArea(text_coordinate_np)
            if small_area == 0:
                text_coordinates.append(text_coordinate)
                continue

            found = False

            for key, value in caret_dict.items():
                caret_coordinate_np = np.array(value[0], dtype=np.float32)

                inside = cv2.pointPolygonTest(caret_coordinate_np, tuple(text_coordinate_np[0]), False)
                if inside < 0:
                    if not any(cv2.pointPolygonTest(caret_coordinate_np, tuple(pt), False) >= 0 for pt in text_coordinate_np):
                        continue

                retval, _ = cv2.intersectConvexConvex(text_coordinate_np, caret_coordinate_np)
                intersect_area = retval if retval > 0 else 0
                overlap_ratio = intersect_area / small_area

                if overlap_ratio >= overlap_thresh:
                    caret_text_coordinates.append(text_coordinate)
                    caret_dict[key].append(text_coordinate)
                    found = True
                    break

            if not found:
                text_coordinates.append(text_coordinate)

        return text_coordinates, caret_text_coordinates

    def text_detection(self, args, text_model, image, output_path, filename, device):
        boxes = []
        image_copy = image.copy()
        image_np = np.array(image_copy) 

        results = text_model(image_np, imgsz=args.text_size, max_det=2000, iou=0.3, device=device, verbose=False)
        
        result = results[0]
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            coords = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            boxes.append(coords)

            cv2.rectangle(image_np, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        if args.debug_mode:
            text_path = os.path.join(output_path, f'{filename}_text.jpg')
            cv2.imwrite(text_path, image_np)

        return boxes
    
    def sort_textbox(self, args, textbox_coordinate):
        sort_textbox = []
        while(len(textbox_coordinate)>0):
            polys = sorted(textbox_coordinate, key=lambda quad: quad[1][0], reverse=True)[:args.column]
            polys = sorted(polys, key=lambda quad: quad[0][1], reverse=False)
            for poly in polys:
                sort_textbox.append(poly)
            textbox_coordinate = [item for item in textbox_coordinate 
                        if not any(np.array_equal(item, p) for p in polys)]

        return sort_textbox

    def ignore_text_detection(self, args, ignore_model, image, output_path, filename, device):
        results = ignore_model(image, imgsz=args.ignore_size, max_det=1, device=device, verbose=False)
        
        result = results[0]
        if len(result.boxes) != 0:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)  
                x1, y1, x2, y2 = xyxy

            ref_point = (x2, y1)
            ref_color = image[ref_point[1], ref_point[0]] 

            polygon = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)

            center = np.mean(polygon, axis=0) 

            scale = 1
            scaled_polygon = (polygon - center) * scale + center  

            scaled_polygon_int = scaled_polygon.astype(np.int32)

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [scaled_polygon_int], 255)

            image[mask == 255] = ref_color
            
            if args.debug_mode:
                cv2.imwrite(os.path.join(output_path, "ignore.png"), image)

        return image

    def caret_mark_detection(self, args, caret_mark_model, caret_dict, caret_image_dict, output_path, device):
        if args.debug_mode:
            caret_mark_output = os.path.join(output_path, args.caret_mark_output)
            if not os.path.exists(caret_mark_output):
                os.makedirs(caret_mark_output)

        for idx, (key, value) in enumerate(caret_image_dict.items()):
            filename = key
            image = value
            draw_image = image.copy()

            results = caret_mark_model(image, imgsz=args.caret_mark_size, device=device, verbose=False, max_det=1)
            result = results[0]

            if len(result.boxes) != 0:
                for box in result.boxes:
                    conf = box.conf[0].item()
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy

                    caret_top_left_coordinate = caret_dict[filename][0][0] 

                    caret_mark_coordinate = [
                        [x1 + caret_top_left_coordinate[0], y1 + caret_top_left_coordinate[1]],
                        [x2 + caret_top_left_coordinate[0], y1 + caret_top_left_coordinate[1]],
                        [x2 + caret_top_left_coordinate[0], y2 + caret_top_left_coordinate[1]],
                        [x1 + caret_top_left_coordinate[0], y2 + caret_top_left_coordinate[1]]
                    ]

                    caret_dict[filename].append(caret_mark_coordinate)

                    cv2.rectangle(draw_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

                if args.debug_mode:
                    caret_path = os.path.join(caret_mark_output, f'{filename}.jpg')
                    cv2.imwrite(caret_path, draw_image)
            else:
                del caret_dict[filename]

        return caret_dict

    def caret_detection(self, args, caret_model, image, output_path, filename, device):
        caret_output = os.path.join(output_path, args.caret_output)

        if args.debug_mode:
            if not os.path.exists(caret_output):
                os.makedirs(caret_output)

        draw_image = image.copy()

        results = caret_model(image, imgsz=args.caret_size, device=device, verbose=False, max_det=2000)

        caret_coordinate = []
        caret_dict = {}
        caret_image_dict = {}
        result = results[0]

        for i, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            coords = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            caret_coordinate.append(coords)
            caret_dict[f'image{i}.jpg'] = [coords]

            crop_caret = image[y1:y2, x1:x2]
            caret_image_dict[f'image{i}.jpg'] = crop_caret

            if args.debug_mode:
                crop_caret_path = os.path.join(caret_output, f'image{i}.jpg')
                cv2.imwrite(crop_caret_path, crop_caret)

            cv2.rectangle(draw_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        if args.debug_mode:
            caret_path = os.path.join(output_path, f'{filename}_caret.jpg')
            cv2.imwrite(caret_path, draw_image)

        return caret_dict, caret_image_dict
    
    def papper_stretch(self, args, image, sort_textbox_coordinate, output_path, filename):
        W, H = image.shape[1], image.shape[0]

        corners_img = {
            "top_left": (0, 0),
            "top_right": (W, 0),
            "bottom_left": (0, H),
            "bottom_right": (W, H)
        }

        all_points = [pt for poly in sort_textbox_coordinate for pt in poly]

        closest_points = {}
        for name, (cx, cy) in corners_img.items():
            distances = [math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in all_points]
            idx = distances.index(min(distances))
            closest_points[name] = all_points[idx]

        src_pts = np.array([
            closest_points["top_left"],
            closest_points["top_right"],
            closest_points["bottom_right"],
            closest_points["bottom_left"]
        ], dtype=np.float32)

        dst_pts = np.array([
            [0, 0],
            [W, 0],
            [W, H],
            [0, H]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(image, M, (W, H))

        if args.debug_mode:
            output_path = os.path.join(output_path, f'{filename}_wrap.jpg')
            cv2.imwrite(output_path, warped)

        return warped
    
    def get_island_distance(self, island_a, island_b):
        min_dist = float('inf')
        
        centers_a = [np.mean(np.array(box), axis=0) for box in island_a]
        centers_b = [np.mean(np.array(box), axis=0) for box in island_b]
        
        centers_a = np.array(centers_a)
        centers_b = np.array(centers_b)
        
        dists = np.linalg.norm(centers_a[:, None] - centers_b, axis=2)
        
        return np.min(dists)

    def merge_islands_to_target(self, boxes, target_count, strict_threshold_ratio=1.0, max_merge_ratio=3.0):
        if not boxes:
            return []
        
        n = len(boxes)
        if n < 2: return boxes

        centers = []
        diagonal_lengths = []
        for box in boxes:
            box_np = np.array(box)
            centers.append(np.mean(box_np, axis=0))
            w = np.max(box_np[:, 0]) - np.min(box_np[:, 0])
            h = np.max(box_np[:, 1]) - np.min(box_np[:, 1])
            diagonal_lengths.append(np.sqrt(w**2 + h**2))

        avg_diagonal = np.mean(diagonal_lengths)
        strict_threshold = avg_diagonal * strict_threshold_ratio
        stop_threshold = avg_diagonal * max_merge_ratio

        adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < strict_threshold:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = set()
        all_islands = []
        
        for i in range(n):
            if i not in visited:
                current_indices = []
                queue = [i]
                visited.add(i)
                while queue:
                    node = queue.pop(0)
                    current_indices.append(node)
                    for neighbor in adj[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                island_boxes = [boxes[idx] for idx in current_indices]
                all_islands.append(island_boxes)

        print(len(all_islands))

        if not all_islands: return boxes

        all_islands.sort(key=len, reverse=True)
        main_island = all_islands[0]
        candidate_islands = all_islands[1:]

        while len(main_island) < target_count and len(candidate_islands) > 0:
            best_dist = float('inf')
            best_island_idx = -1
            
            for i, island in enumerate(candidate_islands):
                dist = self.get_island_distance(main_island, island)
                if dist < best_dist:
                    best_dist = dist
                    best_island_idx = i
            
            if best_dist > stop_threshold:
                break
                
            if len(main_island) + len(candidate_islands[best_island_idx]) > target_count:
                candidate_islands.pop(best_island_idx)
                continue

            main_island.extend(candidate_islands[best_island_idx])
            
            candidate_islands.pop(best_island_idx)

        return main_island
    
    def stretch_textbox_detection(self, args, empty_model, image, output_path, filename, device):
        boxes = []
        image = image.copy()
        image_np = np.array(image) 

        results = empty_model(image_np, imgsz=args.textbox_size, max_det=2000, iou=0.4, device=device, verbose=False)
        
        result = results[0]
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            boxes.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

        target_grid_count = args.column * args.row

        boxes = self.merge_islands_to_target(boxes, target_grid_count, strict_threshold_ratio=1.0, max_merge_ratio=2.0)

        sort_textbox_coordinate = self.sort_textbox(args, boxes)

        if args.debug_mode:
            for i, poly in enumerate(sort_textbox_coordinate, 1):

                poly_pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                poly_draw = poly_pts.reshape((-1, 1, 2))
                cv2.polylines(image_np, [poly_draw], isClosed=True, color=(0, 0, 255), thickness=2)

                if len(poly_pts) > 1:
                    x, y = int(poly_pts[1][0]) - 3, int(poly_pts[1][1]) + 3
                else:
                    x, y = int(poly_pts[0][0]), int(poly_pts[0][1])
                cv2.putText(image_np, str(i), (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)

            text_path = os.path.join(output_path, f'{filename}_stretch_textbox.jpg')
            cv2.imwrite(text_path, image_np)

        return sort_textbox_coordinate
    
    def textbox_detection(self, args, empty_model, image, output_path, filename, device):
        boxes = []
        image = image.copy()
        image_np = np.array(image) 

        results = empty_model(image_np, imgsz=args.textbox_size, max_det=2000, iou=0.4, device=device, verbose=False)
        
        result = results[0]
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            boxes.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

        sort_textbox_coordinate = self.sort_textbox(args, boxes)

        if args.debug_mode:
            for i, poly in enumerate(sort_textbox_coordinate, 1):

                poly_pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                poly_draw = poly_pts.reshape((-1, 1, 2))
                cv2.polylines(image_np, [poly_draw], isClosed=True, color=(0, 0, 255), thickness=2)

                if len(poly_pts) > 1:
                    x, y = int(poly_pts[1][0]) - 3, int(poly_pts[1][1]) + 3
                else:
                    x, y = int(poly_pts[0][0]), int(poly_pts[0][1])
                cv2.putText(image_np, str(i), (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)

            text_path = os.path.join(output_path, f'{filename}_textbox.jpg')
            cv2.imwrite(text_path, image_np)

        return sort_textbox_coordinate

    def phone_papper_stretch(self, args, phone_papper_model, image, output_path, filename, device):
        def get_nearest_points(mask, rect_pts):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return []

            all_points = np.vstack(contours).squeeze()
            nearest_pts = []
            for pt in rect_pts:
                dists = distance.cdist([pt], all_points)
                nearest = all_points[np.argmin(dists)]
                nearest_pts.append(tuple(nearest))
            return nearest_pts

        def order_points(pts):
            pts = np.array(pts, dtype="float32")
            rect = np.zeros((4, 2), dtype="float32")

            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  
            rect[2] = pts[np.argmax(s)]  

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  
            rect[3] = pts[np.argmax(diff)]  

            return rect
        
        results = phone_papper_model(image, imgsz=args.phone_papper_size, max_det=1, device=device, verbose=False)
        result = results[0]

        masks = result.masks.data.cpu().numpy()
        mask = masks[0]
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8) * 255
        if args.debug_mode:
            cv2.imwrite(os.path.join(output_path, "mask.png"), mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        nearest_pts = get_nearest_points(mask, box)

        src_pts = order_points(nearest_pts)

        widthA = np.linalg.norm(src_pts[2] - src_pts[3])
        widthB = np.linalg.norm(src_pts[1] - src_pts[0])
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(src_pts[1] - src_pts[2])
        heightB = np.linalg.norm(src_pts[0] - src_pts[3])
        maxHeight = int(max(heightA, heightB))

        dst_pts = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        vis = image.copy()
        cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
        for i, pt in enumerate(nearest_pts):
            cv2.circle(vis, pt, 5, (0, 0, 255), -1)
            cv2.putText(vis, f"P{i+1}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        if args.debug_mode:
            cv2.imwrite(os.path.join(output_path, "box.png"), vis)
            cv2.imwrite(os.path.join(output_path, "warped.png"), warped)

        return warped

    def rotate_image(self, args, angle_model, image, device):
        results = angle_model(image, imgsz=args.angle_size, device=device, verbose=False)

        for result in results:
            probs = result.probs  
            pred_idx = probs.top1  
            confidence = probs.data[pred_idx].item()  
            class_name = angle_model.names[pred_idx]  

        image = np.array(image)

        if class_name == 'rotate0':
            pass
        elif class_name == 'rotate90':
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif class_name == 'rotate180':
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif class_name == 'rotate270':
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        return image
    
    def load_image_path(self, folder_path):
        image_paths = []
        for image_name in os.listdir(folder_path):
            if image_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(folder_path, image_name)
                image_paths.append(image_path)

        return image_paths

    def load_model(self, args, project_root):
        print('Load model...')

        angle_model = YOLO(project_root/args.angle_weight)

        phone_papper_model = YOLO(project_root/args.phone_papper_weight)

        caret_model = YOLO(project_root/args.caret_weight)

        caret_mark_model = YOLO(project_root/args.caret_mark_weight)

        ignore_model = YOLO(project_root/args.ignore_weight)

        text_model = YOLO(project_root/args.text_weight)

        textbox_model = YOLO(project_root/args.textbox_weight)

        return angle_model, phone_papper_model, caret_model, caret_mark_model, ignore_model, text_model, textbox_model

    def load_papper_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return data

    def get_argparse(self):
        parser = argparse.ArgumentParser(description='Text Detection')
        #稿紙參數檔路徑
        parser.add_argument('--papper_config', default=r'./config/papper_config.yaml', type=str, help='稿紙參數檔路徑')

        #模型權重路徑
        parser.add_argument('--angle_weight', default=r'./weight/angle.pt', type=str, help='旋轉分類模型權重')
        parser.add_argument('--phone_papper_weight', default=r'./weight/phone_papper.pt', type=str, help='稿紙分割模型權重')
        parser.add_argument('--caret_weight', default=r'./weight/caret.pt', type=str, help='插入字符檢測模型權重')
        parser.add_argument('--caret_mark_weight', default=r'./weight/caret_mark.pt', type=str, help='插入符號檢測模型權重')
        parser.add_argument('--ignore_weight', default=r'./weight/ignore.pt', type=str, help='忽略字檢測模型權重')
        parser.add_argument('--text_weight', default=r'./weight/text.pt', type=str, help='文字檢測模型權重')
        parser.add_argument('--textbox_weight', default=r'./weight/textbox.pt', type=str, help='空白檢測模型權重')

        #測試模式閾值
        parser.add_argument('--resoultion_threshold', default=23, type=int, help='切割文字解析度閾值(若test_mode為True則會將切割文字平均解析度低於閾值的影像過濾不做)') 

        #模型參數
        parser.add_argument("--angle_size", type=int, default=1280, help="旋轉分類影像縮放尺寸")
        parser.add_argument("--phone_papper_size", type=int, default=960, help="稿紙分割影像縮放尺寸")
        parser.add_argument("--caret_size", type=int, default=1280, help="插入字符檢測影像縮放尺寸")
        parser.add_argument("--caret_mark_size", type=int, default=64, help="插入符號檢測影像縮放尺寸")
        parser.add_argument('--ignore_size', default=1280, type=int, help='忽略字檢測影像縮放尺寸') 
        parser.add_argument('--text_size', default=1280, type=int, help='文字檢測影像縮放尺寸') 
        parser.add_argument("--textbox_size", type=int, default=1280, help="文字框檢測影像縮放尺寸")
        parser.add_argument("--empty_size", type=int, default=1280, help="空白檢測影像縮放尺寸")

        #影像輸入/輸出路徑
        parser.add_argument('--output', default=r'./output', type=str, help='預測輸出路徑') 
        parser.add_argument('--text_output', default='text', type=str, help='文字檢測輸出路徑') 
        parser.add_argument('--caret_output', default='caret', type=str, help='插入字符檢測輸出路徑') 
        parser.add_argument('--caret_mark_output', default='caret mark', type=str, help='插入符號檢測輸出路徑') 
        parser.add_argument('--text_textbox_split_output', default='text textbox split', type=str, help='切割文字集空白輸出路徑') 
        parser.add_argument('--split_output', default='split', type=str, help='切割文字輸出路徑') 
        parser.add_argument('--post_procss_split_output', default='post process split', type=str, help='後處理切割文字輸出路徑') 

        return parser.parse_args()

    def predict(self, image_list, image_amount, image_path_list, device):
        response_image_base64_list = []
        current_dir = Path(__file__).resolve().parent 
        project_root = current_dir.parents[2]  

        args = self.get_argparse()
        data = self.load_papper_config(project_root/args.papper_config)

        args.column = data['column']
        args.row = data['row']
        args.example_format = data['example format']
        args.high_school_format = data['high school format']
        args.test_mode = data['test mode']
        args.debug_mode = data['debug_mode']
        args.noise = data['noise']
        
        angle_model, phone_papper_model, caret_model, caret_mark_model, ignore_model, text_model, textbox_model = self.load_model(args, project_root)
        
        for image_index, image in enumerate(image_list):
            image_path = image_path_list[image_index]
            filename, file_ext = os.path.splitext(os.path.basename(image_path))

            output_root = os.path.join(project_root, args.output)
            output_path = os.path.join(project_root/args.output, filename)
            text_textbox_split_path = os.path.join(project_root/args.output, filename, args.text_textbox_split_output)
            split_path = os.path.join(project_root/args.output, filename, args.split_output)
            post_process_split_path = os.path.join(project_root/args.output, filename, args.post_procss_split_output)
            
            if args.debug_mode:
                if not os.path.exists(output_root):
                    os.makedirs(output_root)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                if not os.path.exists(text_textbox_split_path):
                    os.makedirs(text_textbox_split_path)
                if not os.path.exists(split_path):
                    os.makedirs(split_path)
                if not os.path.exists(post_process_split_path):
                    os.makedirs(post_process_split_path)

            if len(image.shape) == 2 or image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            try:
                if args.example_format:
                    image = self.rotate_image(args, angle_model, image, device)
                    image = self.phone_papper_stretch(args, phone_papper_model, image, output_path, filename, device)
                    image = self.ignore_text_detection(args, ignore_model, image, output_path, filename, device)
                else:
                    sort_textbox_coordinate = self.stretch_textbox_detection(args, textbox_model, image, output_path, filename, device)
                    image = self.papper_stretch(args, image, sort_textbox_coordinate, output_path, filename)

                caret_dict, caret_image_dict = self.caret_detection(args, caret_model, image, output_path, filename, device)

                caret_dict = self.caret_mark_detection(args, caret_mark_model, caret_dict, caret_image_dict, output_path, device)

                sort_textbox_coordinate = self.textbox_detection(args, textbox_model, image, output_path, filename, device)

                all_text_coordinate = self.text_detection(args, text_model, image, output_path, filename, device)

                text_coordinate, caret_text_coordinate = self.filter_coordinate(all_text_coordinate, caret_dict)

                image_base64_list = self.saveResult(args, image_index, image_amount, filename, image, sort_textbox_coordinate, text_coordinate, caret_dict, output_path, split_path, text_textbox_split_path, post_process_split_path)

                response_image_base64_list.append(image_base64_list)

            except Exception as e:
                response = OrderedDict({
                    'success': False,
                    'response_image_base64_list': None
                })
            
        response = OrderedDict({
            'success': True,
            'response_image_base64_list': response_image_base64_list
        })

        return response
