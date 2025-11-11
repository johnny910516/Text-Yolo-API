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

    def split_postprocesser(self, image_list, post_process_split_path):
        index = 0
        post_process_image_path_list = []

        for pil_image in image_list:
            if isinstance(pil_image, str):
                post_process_image_path_list.append(pil_image)
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

            post_process_image_path_list.append(f"{post_process_split_path}/{name}.jpg")

        return post_process_image_path_list

    def check_split_word_resolution(self, image_list):
        resolution_list = []
        for image in image_list:
            if isinstance(image, str):
                continue
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

    def search_caret_coordinate_insert_position(self, insert_caret_point_image, new_sort_text_coordinate, caret_dict, insert_caret_point_path):
        def caret_mark_direction_detection(coordinate_list):
            if coordinate_list[1] is not None:
                big_left = np.min(coordinate_list[0][:, 0])
                big_right = np.max(coordinate_list[0][:, 0])

                small_center_x = np.mean(coordinate_list[1][:, 0])

                dist_to_left = abs(small_center_x - big_left)
                dist_to_right = abs(small_center_x - big_right)

                return "right" if dist_to_left < dist_to_right else "left"
            else:
                if len(coordinate_list) > 2:
                    big_left = np.min(coordinate_list[0][:, 0])
                    big_right = np.max(coordinate_list[0][:, 0])

                    small_center_x = np.mean(coordinate_list[2][:, 0])

                    dist_to_left = abs(small_center_x - big_left)
                    dist_to_right = abs(small_center_x - big_right)

                    return "right" if dist_to_left < dist_to_right else "left"
                
                return None

        for key, value in caret_dict.items():
            caret_text_coordinates = [value[i] for i in range(2, len(value))]
            caret_mark_direction = caret_mark_direction_detection(value)

            if caret_mark_direction == 'right':
                caret_text_coordinates = sorted(caret_text_coordinates, key=lambda quad: quad[0][1], reverse=False)
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

                    poly = np.array(poly)

                    br_x = np.max(poly[:, 0])
                    br_y = np.max(poly[:, 1])
                    br_point = (br_x, br_y)

                    dist = np.linalg.norm(np.array([x_insert_position, y_insert_position]) - np.array([br_x, br_y]))
                    if dist < min_dist:
                        min_dist = dist
                        insert_poly = poly
                        insert_index = i
                
                if y_insert_position < insert_poly[0][1]:
                    insert_index -= 1

                for caret_text_coordinate in caret_text_coordinates:
                    insert_index += 1
                    new_sort_text_coordinate.insert(insert_index, np.array(caret_text_coordinate).astype(np.int32))

            elif caret_mark_direction == 'left':
                caret_text_coordinates = sorted(caret_text_coordinates, key=lambda quad: quad[0][1], reverse=False)
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

                    poly = np.array(poly)

                    bl_x = np.min(poly[:, 0])  
                    bl_y = np.max(poly[:, 1])  
                    bl_point = (bl_x, bl_y)

                    dist = np.linalg.norm(np.array([x_insert_position, y_insert_position]) - np.array([bl_x, bl_y]))
                    if dist < min_dist:
                        min_dist = dist
                        insert_poly = poly
                        insert_index = i

                if y_insert_position < insert_poly[0][1]:
                    insert_index -= 1

                for caret_text_coordinate in caret_text_coordinates:
                    insert_index += 1
                    new_sort_text_coordinate.insert(insert_index, np.array(caret_text_coordinate).astype(np.int32))

        cv2.imwrite(insert_caret_point_path, insert_caret_point_image)
            
        return new_sort_text_coordinate
    
    def poly_to_bbox(self, poly):
        xs, ys = poly[:,0], poly[:,1]
        return [xs.min(), ys.min(), xs.max(), ys.max()]
    
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

        # 建立 IoU 矩陣
        iou_matrix = np.zeros((len(A), len(B)))
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                iou_matrix[i, j] = self.iou(a, b)

        # 貪婪匹配
        while True:
            if np.all(iou_matrix == -1):
                break

            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[i, j]

            matches.append((i, j))
            unmatched_A_idx.discard(i)
            unmatched_B_idx.discard(j)

            # 標記為已匹配
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1

        # 回傳：匹配結果 + 未匹配索引
        return list(unmatched_A_idx), list(unmatched_B_idx)
    
    def high_school_insert_paragraph_mark(self, args, image_index, image_amount, sort_text_coordinate, sort_textbox_coordinate):
        new_sort_text_coordinate = []
        text_textbox_coordinate = []
        post_unmatched_A_idx = []
        empty_column_amount = 0

        for i in range(args.row):
            try:
                textbox_list = sort_textbox_coordinate[args.column*i:args.column*(i+1)]
                text_list = sort_text_coordinate[:args.column+5]

                all_A_points = np.vstack(textbox_list) 
                min_x, min_y = np.min(all_A_points, axis=0)
                max_x, max_y = np.max(all_A_points, axis=0)

                points_in_range = []
                for text in text_list:
                    poly_pts = np.array(text)
                    if np.all((poly_pts[:,0] >= min_x-10) & (poly_pts[:,0] <= max_x+10)):  # & (poly_pts[:,1] >= min_y-10) & (poly_pts[:,1] <= max_y+10)
                        points_in_range.append(text)

                sort_text_coordinate = [a for a in sort_text_coordinate if not any(np.array_equal(a, b) for b in points_in_range)]

                unmatched_A_idx, unmatched_B_idx = self.match_iou_max_no_threshold(textbox_list, points_in_range)

                tmp_unmatched_A_idx = unmatched_A_idx.copy()
                if 0 in tmp_unmatched_A_idx and 1 in tmp_unmatched_A_idx:
                    tmp_unmatched_A_idx = [x for x in tmp_unmatched_A_idx if x not in (0, 1)]
                
                # 如果整個直排空白
                if set(unmatched_A_idx) == set(range(args.column)) and len(sort_text_coordinate) == 0:
                    new_sort_text_coordinate.extend(['*', '*']) 
                    break
                elif set(unmatched_A_idx) == set(range(args.column)) and len(sort_text_coordinate) != 0:
                    empty_column_amount+=1
                    if empty_column_amount == 1 and not isinstance(new_sort_text_coordinate[-1], str):
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

        #如果整張影像寫滿
        if image_index == (image_amount-1) and i == args.row-1 and not isinstance(new_sort_text_coordinate[-1], str) and not isinstance(new_sort_text_coordinate[-2], str):
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
                textbox_list = sort_textbox_coordinate[args.column*i:args.column*(i+1)]
                text_list = sort_text_coordinate[:args.column+5]

                all_A_points = np.vstack(textbox_list) 
                min_x, min_y = np.min(all_A_points, axis=0)
                max_x, max_y = np.max(all_A_points, axis=0)

                points_in_range = []
                for text in text_list:
                    poly_pts = np.array(text)
                    if np.all((poly_pts[:,0] >= min_x-10) & (poly_pts[:,0] <= max_x+10)):  # & (poly_pts[:,1] >= min_y-10) & (poly_pts[:,1] <= max_y+10)
                        points_in_range.append(text)

                sort_text_coordinate = [a for a in sort_text_coordinate if not any(np.array_equal(a, b) for b in points_in_range)]

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
                    flag2  = True
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

        #如果整張影像寫滿
        if image_index == (image_amount-1) and i == args.row-1 and not isinstance(new_sort_text_coordinate[-1], str) and not isinstance(new_sort_text_coordinate[-2], str):
            new_sort_text_coordinate.extend(['*', '*']) 

        return new_sort_text_coordinate, text_textbox_coordinate
    
    def sort_text(self, args, sort_textbox_coordinate, text_coordinates):
        sort_text_coordinate = []

        for i in range(args.row):
            text_coordinates = sorted(text_coordinates, key=lambda quad: quad[1][0], reverse=True)
            textbox_list = sort_textbox_coordinate[args.column*i:args.column*(i+1)]
            text_list = text_coordinates[:args.column+20]

            all_A_points = np.vstack(textbox_list) 
            min_x, min_y = np.min(all_A_points, axis=0)
            max_x, max_y = np.max(all_A_points, axis=0)

            points_in_range = []
            for text in text_list:
                poly_pts = np.array(text)
                if np.all((poly_pts[:,0] >= min_x-10) & (poly_pts[:,0] <= max_x+10)): #& (poly_pts[:,1] >= min_y-10) & (poly_pts[:,1] <= max_y+10)
                    points_in_range.append(text)

            unmatched_A_idx, unmatched_B_idx = self.match_iou_max_no_threshold(textbox_list, points_in_range)
            text_coordinates = [a for a in text_coordinates if not any(np.array_equal(a, b) for b in points_in_range)]
            points_in_range = [b for i, b in enumerate(points_in_range) if i not in unmatched_B_idx]

            points_in_range = sorted(points_in_range, key=lambda quad: quad[0][1], reverse=False)
            sort_text_coordinate.extend(points_in_range)

        return sort_text_coordinate

    def saveResult(self, args, image_index, image_amount, image_path, image, sort_textbox_coordinate, text_coordinate, caret_dict, output_path, split_path, text_textbox_split_path, post_process_split_path):
        draw_text_image = image.copy()
        insert_caret_point_image = image.copy()

        filename, file_ext = os.path.splitext(os.path.basename(image_path))

        insert_caret_point_path = os.path.join(output_path, f'{filename}_insert_caret_point.jpg')
        text_path = os.path.join(output_path, f'sort_{filename}_text.jpg')

        text_coordinates = []

        for i, coordinate in enumerate(text_coordinate):
            coordinate = np.array(coordinate).astype(np.int32).reshape((-1))
            coordinate = coordinate.reshape(-1, 2)
            text_coordinates.append(coordinate)

        sort_text_coordinate = self.sort_text(args, sort_textbox_coordinate, text_coordinates)

        if args.high_school_format:
            new_sort_text_coordinate, text_textbox_coordinates = self.high_school_insert_paragraph_mark(args, image_index, image_amount, sort_text_coordinate, sort_textbox_coordinate)
        else:
            new_sort_text_coordinate, text_textbox_coordinates = self.insert_paragraph_mark(args, image_index, image_amount, sort_text_coordinate, sort_textbox_coordinate)

        text_textbox_coordinate = []

        for i, coordinate in enumerate(text_textbox_coordinates):
            coordinate = np.array(coordinate).astype(np.int32).reshape((-1))
            coordinate = coordinate.reshape(-1, 2)
            text_textbox_coordinate.append(coordinate)

        text_coordinate = self.search_caret_coordinate_insert_position(insert_caret_point_image, new_sort_text_coordinate, caret_dict, insert_caret_point_path)

        text_textbox_coordinate = self.search_caret_coordinate_insert_position(insert_caret_point_image, text_textbox_coordinate, caret_dict, insert_caret_point_path)

        split_image = image.copy()
        image_list = self.split(text_coordinate, split_image, split_path)

        _ = self.split(text_textbox_coordinate, split_image, text_textbox_split_path)

        post_process_image_path_list = self.split_postprocesser(image_list, post_process_split_path)

        for i, poly in enumerate(text_coordinate, 1):
            if isinstance(poly, str):
                continue
            cv2.polylines(draw_text_image, [poly.reshape((-1, 1, 2))], True, color=(255, 0, 0), thickness=2)
            cv2.putText(draw_text_image, str(i), (poly[1][0] - 3, poly[1][1] + 3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)

        cv2.imwrite(text_path, draw_text_image)

        return post_process_image_path_list

    def filter_coordinate(self, all_text_coordinate, caret_dict, overlap_thresh=0.8):
        caret_text_coordinates = []
        text_coordinates = []

        for text_coordinate in all_text_coordinate:
            text_coordinate_poly = np.array(text_coordinate, dtype=np.float32)

            small_area = cv2.contourArea(text_coordinate_poly)
            if small_area == 0:
                text_coordinates.append(text_coordinate)
                continue

            found = False

            for key, value in caret_dict.items():
                caret_coordinate = np.array(value[0], dtype=np.float32)

                inside = cv2.pointPolygonTest(caret_coordinate, tuple(text_coordinate_poly[0]), False)
                if inside < 0: 
                    if not any(cv2.pointPolygonTest(caret_coordinate, tuple(pt), False) >= 0 for pt in text_coordinate_poly):
                        continue 

                retval, intersection = cv2.intersectConvexConvex(text_coordinate_poly, caret_coordinate)
                intersect_area = retval if retval > 0 else 0

                overlap_ratio = intersect_area / small_area

                if overlap_ratio >= overlap_thresh:
                    caret_text_coordinates.append(text_coordinate)
                    caret_dict[key].append(text_coordinate)
                    found = True
                    break  

            if not found:
                text_coordinates.append(text_coordinate)

        return np.array(text_coordinates, dtype=np.float32), np.array(caret_text_coordinates, dtype=np.float32)

    def text_detection(self, args, text_model, image, output_path, filename, device):
        boxes = []
        image = image.copy()

        results = text_model(image, imgsz=args.text_size, max_det=2000, iou=0.3, device=device, verbose=False)
        
        result = results[0]
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int) 
            x1, y1, x2, y2 = xyxy
            boxes.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        text_path = os.path.join(output_path, f'{filename}_text.jpg')
        cv2.imwrite(text_path, image)

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

    def ignore_text_detection(self, args, ignore_model, image, filename, device):
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

            cv2.imwrite(os.path.join(args.output, filename, "ignore.png"), image)

        return image

    def caret_mark_detection(self, args, caret_mark_model, caret_dict, caret_output, output_path, device):
        caret_mark_output = os.path.join(output_path, args.caret_mark_output)
        if not os.path.exists(caret_mark_output):
            os.makedirs(caret_mark_output)

        image_paths = self.load_image_path(caret_output)
        for i, image_path in enumerate(image_paths):
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            image = cv2.imread(image_path)
            draw_image = image.copy()

            results = caret_mark_model(image, imgsz=args.caret_mark_size, device=device, verbose=False, max_det=1)
            
            result = results[0]

            if len(result.boxes) != 0:
                for box in result.boxes:
                    conf = box.conf[0].item()
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)  
                    x1, y1, x2, y2 = xyxy

                    caret_top_left_coordinate = caret_dict[os.path.basename(image_path)][0][0]
                    caret_mark_coordinate = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32) + caret_top_left_coordinate
                    caret_dict[os.path.basename(image_path)].append(caret_mark_coordinate)

                    cv2.rectangle(draw_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

                caret_path = os.path.join(caret_mark_output, f'{filename}.jpg')
                cv2.imwrite(caret_path, draw_image)
            else:
                caret_dict[os.path.basename(image_path)].append(None)
            
        return caret_dict

    def caret_detection(self, args, caret_model, image, output_path, filename, device):
        caret_output = os.path.join(output_path, args.caret_output)
        if not os.path.exists(caret_output):
            os.makedirs(caret_output)

        draw_image = image.copy()
        results = caret_model(image, imgsz=args.caret_size, device=device, verbose=False, max_det=2000) 
        
        caret_coordinate = []
        caret_dict = {}
        result = results[0]
        for i, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int) 
            x1, y1, x2, y2 = xyxy

            caret_coordinate.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32))
            caret_dict[f'image{i}.jpg'] = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)]

            crop_caret_path = os.path.join(caret_output, f'image{i}.jpg')
            crop_caret = image[y1:y2, x1:x2]
            cv2.imwrite(crop_caret_path, crop_caret)

            cv2.rectangle(draw_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        caret_path = os.path.join(output_path, f'{filename}_caret.jpg')
        cv2.imwrite(caret_path, draw_image)

        return caret_dict, caret_output
    
    def papper_stretch(self, args, image, sort_textbox_coordinate, filename):
        W, H = image.shape[1], image.shape[0]

        corners_img = {
            "top_left": (0, 0),
            "top_right": (W, 0),
            "bottom_left": (0, H),
            "bottom_right": (W, H)
        }

        all_points = np.concatenate(sort_textbox_coordinate, axis=0)

        closest_points = {}
        for name, (cx, cy) in corners_img.items():
            distances = np.sqrt((all_points[:, 0] - cx) ** 2 + (all_points[:, 1] - cy) ** 2)
            idx = np.argmin(distances)
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

        output_path = os.path.join(args.output, filename, f'{filename}_wrap.jpg')

        cv2.imwrite(output_path, warped)

        return warped
    
    def textbox_detection(self, args, empty_model, image, output_path, filename, device):
        boxes = []
        image = image.copy()

        results = empty_model(image, imgsz=args.textbox_size, max_det=2000, iou=0.4, device=device, verbose=False)
        
        result = results[0]
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int) 
            x1, y1, x2, y2 = xyxy
            boxes.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))

        sort_textbox_coordinate = self.sort_textbox(args, boxes)

        for i, poly in enumerate(sort_textbox_coordinate, 1):
            poly_pts = np.array(poly, dtype=np.int32).reshape(-1, 2) 
            
            # 畫多邊形
            poly_draw = poly_pts.reshape((-1, 1, 2))
            cv2.polylines(image, [poly_draw], isClosed=True, color=(0, 0, 255), thickness=2)
            
            if len(poly_pts) > 1:
                x, y = int(poly_pts[1][0]) - 3, int(poly_pts[1][1]) + 3
            else:
                x, y = int(poly_pts[0][0]), int(poly_pts[0][1])
            cv2.putText(image, str(i), (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)

        text_path = os.path.join(output_path, f'{filename}_textbox.jpg')
        cv2.imwrite(text_path, image)

        return sort_textbox_coordinate

    def phone_papper_stretch(self, args, phone_papper_model, image, filename, device):
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
        cv2.imwrite(os.path.join(args.output, filename, "mask.png"), mask)

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

        cv2.imwrite(os.path.join(args.output, filename, "box.png"), vis)
        cv2.imwrite(os.path.join(args.output, filename, "warped.png"), warped)

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
        parser.add_argument("--textbox_size", type=int, default=1280, help="空白檢測影像縮放尺寸")

        #影像輸入/輸出路徑
        parser.add_argument('--output', default=r'./output', type=str, help='預測輸出路徑') 
        parser.add_argument('--text_output', default='text', type=str, help='文字檢測輸出路徑') 
        parser.add_argument('--caret_output', default='caret', type=str, help='插入字符檢測輸出路徑') 
        parser.add_argument('--caret_mark_output', default='caret mark', type=str, help='插入符號檢測輸出路徑') 
        parser.add_argument('--text_textbox_split_output', default='text textbox split', type=str, help='切割文字集空白輸出路徑') 
        parser.add_argument('--split_output', default='split', type=str, help='切割文字輸出路徑') 
        parser.add_argument('--post_procss_split_output', default='post process split', type=str, help='後處理切割文字輸出路徑') 

        return parser.parse_args()

    def predict(self, image_path, image_index, image_amount, filename, device):
        current_dir = Path(__file__).resolve().parent 
        project_root = current_dir.parents[2]  

        args = self.get_argparse()
        data = self.load_papper_config(project_root/args.papper_config)

        args.column = data['column']
        args.row = data['row']
        args.example_format = data['example format']
        args.high_school_format = data['high school format']
        args.test_mode = data['test mode']
        
        angle_model, phone_papper_model, caret_model, caret_mark_model, ignore_model, text_model, textbox_model = self.load_model(args, project_root)

        output_path = os.path.join(project_root/args.output, filename)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        text_textbox_split_path = os.path.join(project_root/args.output, filename, args.text_textbox_split_output)
        if not os.path.exists(text_textbox_split_path):
            os.makedirs(text_textbox_split_path)

        split_path = os.path.join(project_root/args.output, filename, args.split_output)
        if not os.path.exists(split_path):
            os.makedirs(split_path)

        post_process_split_path = os.path.join(project_root/args.output, filename, args.post_procss_split_output)
        if not os.path.exists(post_process_split_path):
            os.makedirs(post_process_split_path)

        image = cv2.imread(str(project_root/image_path))
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if args.example_format:
            image = self.rotate_image(args, angle_model, image, device)
            image = self.phone_papper_stretch(args, phone_papper_model, image, filename, device)
            image = self.ignore_text_detection(args, ignore_model, image, filename, device)
        else:
            sort_textbox_coordinate = self.textbox_detection(args, textbox_model, image, output_path, filename, device)
            image = self.papper_stretch(args, image, sort_textbox_coordinate, filename)

        caret_dict, caret_output = self.caret_detection(args, caret_model, image, output_path, filename, device)

        caret_dict = self.caret_mark_detection(args, caret_mark_model, caret_dict, caret_output, output_path, device)

        sort_textbox_coordinate = self.textbox_detection(args, textbox_model, image, output_path, filename, device)

        all_text_coordinate = self.text_detection(args, text_model, image, output_path, filename, device)

        text_coordinate, caret_text_coordinate = self.filter_coordinate(all_text_coordinate, caret_dict)

        post_process_image_path_list = self.saveResult(args, image_index, image_amount, image_path, image, sort_textbox_coordinate, text_coordinate, caret_dict, output_path, split_path, text_textbox_split_path, post_process_split_path)

        response = OrderedDict({
            'success': True,
            'post_process_image_path_list': post_process_image_path_list
        })

        return response
