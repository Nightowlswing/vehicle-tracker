from flask import Flask, render_template, request
import random
import numpy as np
import cv2
from detector import Detector

app = Flask(__name__)


def image_to_cv2(byte_image):
    np_arr = np.fromstring(byte_image, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


def image_is_valid(*args, **kwargs):
    return True


def get_random_number():
    ltrs = 'AA AM AI AK BM GG KB ND IG SM BK NR EN'.split()
    return {
        'bbox': [random.randint(0, 4000) for i in range(4)],
        'text': random.choice(ltrs) + ''.join([str(random.randint(0, 10)) for i in range(4)]) + random.choice(ltrs)}


@app.route('/')
def upload_file_page():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if image_is_valid(f):
            image = image_to_cv2(f.read())
        else:
            raise Exception
        d = Detector()
        image_data = d.get_image_data(image)

        # f.save(secure_filename(f.filename))
        return render_template('result.html', plates_text=[plate['text'] for plate in image_data])
    else:
        return 'Hi!'
