# Tetx Yolo API

Tetx Yolo API的Web API服務，稿紙文字檢測、文字辨識功能。

## 輸入資料路徑
輸入影像路徑: ./input

## 輸出資料路徑
輸出影像及文字檔路徑: ./output
輸出切割影像路徑(包含空格): ./output/影像檔名/text textbox split
輸出切割影像路徑(不包含空格): ./output/影像檔名/split

## 注意!!! 注意!!! 注意!!!
每次呼叫Text Yolo API時必須輸入同一份稿紙的影像，
假設同學A的作文寫了兩張稿紙，則必須同時將兩張稿紙
影像一起放入輸入影像路徑的資料夾中進行處理，若不同
份稿紙的影像同時輸入會造成輸出文字檔內容錯誤。

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

### 3. 配置稿紙參數
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

服務將在 `http://localhost:5001` 啟動

## API端點
### POST /predict
處理稿紙圖片，執行文字檢測、文字辨識。

#### 回應格式
```json
{
    "error_code": 0,
    "message": "Processing completed successfully",
}
```

#### 錯誤代碼
- `0`: 處理成功
- `1`: 無效的影像路徑
- `2`: 非影像路徑
- `3`: 處理失敗
- `4`: 未知錯誤