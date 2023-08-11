from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import base64
from flask_cors import CORS

from segment_anything import SamPredictor, sam_model_registry

import matting

app = Flask(__name__)
CORS(app)


def init():
    checkpoint = "model/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device='cpu')
    predictor = SamPredictor(sam)
    return predictor


predictor = init();


@app.route('/', methods=['GET'])
def home():
    return jsonify("OK")


@app.route('/segment_everything_box_model', methods=['POST'])
def process_image():
    image_data = request.data

    pil_image = Image.open(io.BytesIO(image_data))

    np_image = np.array(pil_image)

    predictor.set_image(np_image)

    image_embedding = predictor.get_image_embedding().cpu().numpy()

    result_base64 = base64.b64encode(image_embedding.tobytes()).decode('utf-8')
    result_list = [result_base64]
    return jsonify(result_list)

@app.route('/matting', methods=['POST'])
def matting():
    image_data = request.data
    # 将二进制数据转换为PIL图像对象
    image = Image.open(io.BytesIO(image_data))
    mask = Image.open(io.BytesIO(image_data))

    image_matting = matting.matting(image, mask)

    result_base64 = base64.b64encode(image_matting.tobytes()).decode('utf-8')
    result_list = [result_base64]
    return jsonify(result_list)

if __name__ == '__main__':
    app.run()
