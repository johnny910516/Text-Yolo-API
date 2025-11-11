from Split import app
from flask import request, jsonify
from flask_classful import FlaskView, route
from collections import OrderedDict

from Split.Controller.TextYolo import TextYolo


class TextYoloAPI(FlaskView):
    @route('/split', methods=['POST'])
    def split(self):
        # ✅ 用 JSON 取資料（不再用 request.form）
        data = request.get_json(force=True, silent=True)

        # ✅ 檢查 image_path
        image_path = data.get('image_path', '').strip()
        if not image_path:
            return jsonify({
                "error_code": 1,
                "message": "image path is empty",
                "results": None
            })

        # ✅ 檢查副檔名
        if not image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
            return jsonify({
                "error_code": 2,
                "message": "Not image file",
                "image_path": image_path
            })

        image_index = int(data.get('image_index', 0))
        image_amount = int(data.get('image_amount', 0))
        filename = data.get('filename', '')
        device = data.get('device', 'cpu')

        predictor = TextYolo()

        try:
            response = predictor.predict(image_path, image_index, image_amount, filename, device)

            if response['success']:
                json_response = OrderedDict({
                    "error_code": 0,
                    "message": "split success",
                    "post_process_image_path_list": response['post_process_image_path_list']
                })
                return jsonify(json_response)
            else:
                return jsonify({
                    "error_code": 3,
                    "message": "split error"
                })

        except Exception as e:
            return jsonify({
                "error_code": 4,
                "message": f"Unknown error occurred: {str(e)}"
            })


TextYoloAPI.register(app, route_base='/')

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5003)
