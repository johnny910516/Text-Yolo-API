# Tetx Yolo API
Tetx Yolo API的Web API服務，稿紙文字檢測、文字辨識功能。

## 安裝與設置

### 1. 創建虛擬環境
python version: 3.12.4
```bash
python -m venv Tetx_Yolo_API
```
```bash
Tetx_Yolo_API\Scripts\activate
```

### 2. 安裝依賴
```bash
pip3 install --pre torch torchvision torchaudio --inde`x-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

### 3. 配置稿紙參數檔案
./config/papper_config.yaml

### 4. 啟動API服務
```bash
cd Tetx Yolo API
```
```bash
python start_api.py
```

### 5. 測試API
```bash
python test_api.py
```

## 使用說明
每次呼叫Text Yolo API時必須輸入同一份稿紙的影像，
假設同學A的作文寫了兩張稿紙，則必須同時將兩張稿紙
影像一起放入輸入影像路徑的資料夾中進行處理，若不同
份稿紙的影像同時輸入會造成輸出文字檔內容錯誤。

### 輸入資料路徑
輸入影像路徑: ./input

### 輸出資料路徑
輸出影像及文字檔路徑: ./output
輸出切割影像路徑(包含空格): ./output/影像檔名/text textbox split
輸出切割影像路徑(不包含空格): ./output/影像檔名/split

## split API端點

### Split API服務將在 `http://localhost:5001` 啟動
處理稿紙圖片，執行文字檢測，檢測後進行文字排序即加入對應的段落結尾符號。

#### Splot API輸入參數
- image_base64_list: 轉換為base64二進制表示的影像list
- image_path_list: 影像路徑list
- image_amount: 處理影像總數
- device: 使用cpu或是cuda

split_data = {
    'image_base64_list': image_base64_list,
    'image_path_list': image_path_list,
    'image_amount': image_amount,
    'device': str(DEVICE), 
}

#### Split_API錯誤代碼
- `0`: 切割成功
- `1`: 錯誤的API輸入參數
- `2`: 影像取得錯誤
- `3`: 切割失敗
- `4`: 未知錯誤

#### Split_API回應格式(result為經過排序後的文字影像轉換成base64的list，且加入段落符號)
split_response = {
    "error_code": 0,
    "message": "split success",
    "result": ["base64的影像", "base64的影像" ,"#","#", ......, "*", "*"] 
}

## Recognize API端點

#### Recognize API服務將在 `http://localhost:5002` 啟動
處理切割文字影像，執行文字辨識。

#### Splot API輸入參數
- image_base64_list: 切割後且轉換為base64二進制表示的影像list
- image_path_list: 影像路徑list
- image_amount: 處理影像總數
- device: 使用cpu或是cuda

recognize_data = {
    'image_base64_lists': split_response.json()['result'],
    'image_path_list': image_path_list,
    'image_amount': image_amount,
    'device': str(DEVICE)
}

#### Recognize_API錯誤代碼
- `0`: 辨識成功
- `1`: 錯誤的API輸入參數
- `2`: 影像取得錯誤
- `3`: 辨識失敗
- `4`: 未知錯誤

#### Split_API回應格式(result為辨識後的文字list)
- txt_content: 整體辨識文字list
- part1: 高中稿紙第一部分辨識文字list，若非高中稿紙則為None
- part2: 高中稿紙第二部分辨識文字list，若非高中稿紙則為None

recognize_response = {
    "error_code": 0,
    "message": "recognize success",
    "result": {
        'txt_content': ['心', '#', '#', '\n', ......,],
        'part1': '心##\n    紙本書的優點在於閱讀',
        'part2': '心##\n    紙本書的優點在於閱讀'
    }
}