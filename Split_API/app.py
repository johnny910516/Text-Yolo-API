from Split import app
from flask import request, jsonify
from flask_classful import FlaskView, route
from collections import OrderedDict
import base64
import cv2
import numpy as np

from Split.Controller.TextYolo import TextYolo

class TextYoloAPI(FlaskView):
    @route('/split', methods=['POST'])
    def split(self):
        data = request.get_json()
        if not data:
            return jsonify({
                "error_code": 1,
                "message": "No JSON data received"
            })

        image_base64 = data.get('image_base64', '')
        image_index = int(data.get('image_index', 0))
        image_amount = int(data.get('image_amount', 0))
        filename = data.get('filename', '')
        device = data.get('device', 'cpu')

        image_bytes = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                "error_code": 2,
                "message": "No image data received",
                "result": None
            })

        predictor = TextYolo()

        try:
            response = predictor.predict(image, image_index, image_amount, filename, device)

            if response['success']:
                return jsonify({
                    "error_code": 0,
                    "message": "split success",
                    "result": response['image_base64_list']
                })
            else:
                return jsonify({
                    "error_code": 3,
                    "message": "split error",
                    "result": None
                })

        except Exception as e:
            return jsonify({
                "error_code": 4,
                "message": f"Unknown error occurred: {str(e)}",
                "result": None
            })

TextYoloAPI.register(app, route_base='/')

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5001)