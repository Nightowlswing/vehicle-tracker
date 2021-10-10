import logging
import os
import io

from flask import Flask, render_template, request, send_file, redirect
import numpy as np
import cv2
from PIL import Image

from redis_api import add_image, get_image_file, get_image_data, clear_all, get_all_images_ids
from detector import Detector



app = Flask(__name__)
result = None
detector = Detector()
logger = logging.getLogger()


def get_images(file_data):
    with open("video.mp4", "wb") as file:
        file.write(file_data)
    vidcap = cv2.VideoCapture('video.mp4')
    success,image = vidcap.read()
    count = 0
    images = []
    while success:  # save frame as JPEG file      
        success,image = vidcap.read()
        if count%10 == 0:
            images.append(image)
        count += 1
        if count > 4000:
            break
    os.remove("video.mp4")
    return images
        

def image_to_cv2(byte_image):
    image = np.asarray(bytearray(byte_image), dtype="uint8")
    loaded_file = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return loaded_file


def image_is_valid(*args, **kwargs):
    return True


@app.route('/')
def upload_file_page():
    image_ids = get_all_images_ids()
    data = [{"url": f"/image/{i}", "data": get_image_data(i)} for i in image_ids]
    return render_template('index.html', images=data)


@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    global result
    if request.method == 'POST':
        try:
            f = request.files['file']
            if image_is_valid(f):
                images = [image for image in get_images(f.read())]            
            else:
                raise Exception
            clear_all()
            r = []
            for i, image in enumerate(images):
                image_data = detector.get_image_data(image)
                result = detector.draw_on_image(image, image_data)
                add_image(result, ", ".join([im[1] for im in image_data]), i)
                # f.save(secure_filename(f.filename))
            return redirect("/")
        except Exception as e:
            logger.error(f"|upload_file| {e}")
            return redirect("/oops")
    else:
        return 'Hi!'

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            f = request.files['file']
            if image_is_valid(f):
                image = image_to_cv2(f.read())
            else:
                raise Exception
            clear_all()
            image_data = detector.get_image_data(image)
            
            result = detector.draw_on_image(image, image_data)
            add_image(result, ", ".join([im[1] for im in image_data]))
            # f.save(secure_filename(f.filename))
            return redirect("/")
        except Exception as e:
            logger.error(f"|upload_file| {e}")
            return redirect("/oops")
    else:
        return 'Hi!'


@app.route("/image/<image_id>", methods=["GET", "POST"])
def downloader(image_id: int):
    if request.method == "GET":
        try:
            image = get_image_file(image_id)
            result = io.BytesIO(image)
            result.seek(0)
            return send_file(result, mimetype='image/jpeg')
        except Exception as e:
            logger.error(f"|upload_file| {e}")
            return redirect("/oops")
    else:
        return 'Hi!'

@app.route("/oops", methods=["GET"])
def oops():
    return render_template("oops.html")
