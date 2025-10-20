from flask import Flask
from flask import request, jsonify
from flask_classful import FlaskView
from collections import OrderedDict
import numpy as np

from TextYolo import TextYolo

class TextYoloAPI(FlaskView):
    def __init__(self):
        self.txt_content = np.array([''], dtype=object)

    def post(self):
        try:
            if not request.form.get('image_path').strip():
                json_response = OrderedDict({
                    "error_code": 1,
                    "message": "image path is empty",
                    "results": None
                })
                return jsonify(json_response), 400
            
            if not request.form.get('image_path').lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                json_response = OrderedDict({
                    "error_code": 2,
                    "message": "Not image file",
                    "image_path": request.form.get('image_path')
                })
                return jsonify(json_response), 400
            
            image_path = request.form.get('image_path')
            image_index = int(request.form.get('image_index'))
            image_amount = int(request.form.get('image_amount'))
            filename = request.form.get('filename')

            predictor = TextYolo()
            result = predictor.predict(image_path, image_index, image_amount, filename, self.txt_content)

            if result['success']:
                json_response = OrderedDict({
                    "error_code": 0,
                    "message": "Process success",
                })
                return jsonify(json_response)
            else:
                json_response = OrderedDict({
                    "error_code": 3,
                    "message": 'Process error',
                })
                return jsonify(json_response), 500

        except Exception as e:
            json_response = OrderedDict({
                "error_code": 4,
                "message": f"Unknown error occurred: {str(e)}",
            })
            return jsonify(json_response), 500
    
app = Flask(__name__)
TextYoloAPI.register(app, route_base='/post')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
        
        



    

