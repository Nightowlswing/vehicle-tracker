from flask import Flask, render_template, request, send_file, redirect
import random
import numpy as np
from numpy.core.fromnumeric import reshape
import cv2
from detector import Detector

app = Flask(__name__)
result = None
detector = Detector()

def image_to_cv2(byte_image):
    image = np.asarray(bytearray(byte_image), dtype="uint8")
    loaded_file = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return loaded_file


def image_is_valid(*args, **kwargs):
    return True


@app.route('/')
def upload_file_page():
    return render_template('index.html', result=bool(result))


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    global result
    if request.method == 'POST':
        f = request.files['file']
        if image_is_valid(f):
            image = image_to_cv2(f.read())
        else:
            raise Exception
        image_data = detector.get_image_data(image)
        result = detector.draw_on_image(image, image_data)
        # f.save(secure_filename(f.filename))
        return redirect("/")
    else:
        return 'Hi!'

@app.route("/downloader", methods=["GET", "POST"])
def downloader():
    if request.method == "GET":
        return send_file(result, mimetype='image/jpeg')
