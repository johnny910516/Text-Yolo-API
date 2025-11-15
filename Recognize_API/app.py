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
            data = request.get_json()
            if not data:
                return jsonify({
                    "error_code": 1,
                    "message": "No JSON data received",
                    "result": None
                })

            image_base64_list = data.get('image_base64_list', [])
            image_index = int(data.get('image_index', 0))
            image_amount = int(data.get('image_amount', 0))
            filename = data.get('filename', '')
            file_ext = data.get('file_ext', '')
            txt_content = data.get('txt_content', '')
            device = data.get('device', 'cpu')

            if len(image_base64_list) == 0:
                return jsonify({
                    "error_code": 2,
                    "message": "No image data received",
                    "result": None
                })

            predictor = HTRYolo()

            response = predictor.predict(image_amount, image_index, filename, file_ext, image_base64_list, txt_content, device)

            if response['success']:
                return jsonify({
                    "error_code": 0,
                    "message": "recognize success",
                    "result": {
                        "txt_content": response['txt_content'],
                        "part1": response['part1'],
                        "part2": response['part2']
                    }
                })
            else:
                return jsonify({
                    "error_code": 3,
                    "message": "recognize error",
                    "result":None
                })

        except Exception as e:
            return jsonify({
                "error_code": 4,
                "message": f"Unknown error occurred: {str(e)}",
                "result":None
            })
        
HTRYoloAPI.register(app, route_base='/')

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5002)