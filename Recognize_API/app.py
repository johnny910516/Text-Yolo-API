from Recognize import app
from flask import request, jsonify
from flask_classful import FlaskView, route
from collections import OrderedDict
import os

from Recognize.Controller.HTRYolo import HTRYolo

class HTRYoloAPI(FlaskView):
    @route('/recognize', methods=['POST'])
    def recognize(self):
        try:
            print('收到請求 /recognize')
            
            # ✅ 改成接 JSON
            data = request.get_json()
            if not data:
                return jsonify({
                    "error_code": 1,
                    "message": "No JSON data received"
                }), 400

            # 讀取傳入的參數
            image_index = int(data.get('image_index', 0))
            image_amount = int(data.get('image_amount', 0))
            filename = data.get('filename', '')
            file_ext = data.get('file_ext', '')
            post_process_image_path_list = data.get('post_process_image_path_list', [])
            txt_content = data.get('txt_content', '')
            device = data.get('device', 'cpu')

            predictor = HTRYolo()

            # 執行模型
            response = predictor.predict(image_amount, image_index, filename, file_ext, post_process_image_path_list, txt_content, device)

            # 回傳結果
            if response['success']:
                return jsonify({
                    "error_code": 0,
                    "message": "recognize success",
                    "txt_content": response['txt_content']
                })
            else:
                return jsonify({
                    "error_code": 3,
                    "message": "recognize error"
                }), 500

        except Exception as e:
            return jsonify({
                "error_code": 4,
                "message": f"Unknown error occurred: {str(e)}"
            }), 500
        
HTRYoloAPI.register(app, route_base='/')

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5004)