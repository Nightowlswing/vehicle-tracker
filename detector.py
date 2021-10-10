import cv2
import numpy 
from typing import Dict, List, Any, Tuple
import pytesseract
from PIL import Image
import io
import time

from plate_detector import PlateDetector


class Detector:
    """
        class to get image data (bounding boxes of license plates with description)
    """
    def __init__(self):
        self.plateDetector = PlateDetector('weights/best.pt', 'cpu')
        


    def get_image_data(self, original_image: numpy.ndarray) -> List[Dict[str, Any]]:
        """
        get image data (bounding boxes of license plates with description)
        :param image: opencv image
        :return: image data
        """
        start = time.time()
        bounding_boxes = self.plateDetector.get_boxes(original_image)
        print(bounding_boxes)
        # boxes_with_text = [{'box': box, 'text': self._get_text(image, box)} for box in bounding_boxes]
        result = []
        # TODO for i in range(len(bounding_boxes[:])):
        for i in range(len(bounding_boxes[:])):
            box = bounding_boxes[i, :]
            plate = original_image[box[1]:box[3], box[0]:box[2]]
            text = recognize_text(plate)
            result.append((box, text))

        print( "TIME on 1 image boxes", time.time()- start)
        return result

    def draw_on_image(self, image: numpy.ndarray, boxes_with_test: Tuple[numpy.ndarray, str]):
        for box, text in boxes_with_test:
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
            cv2.putText(image, text, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, 'JPEG', quality=70)
        img_byte_arr.seek(0)
        print(id(img_byte_arr))
        return img_byte_arr


def recognize_text(image: numpy.ndarray):
    symbols = format_image(image)
    result = ""
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    for s in symbols:
        
        
        r = pytesseract.image_to_string(s, lang='eng',
                                              config='--oem 3 --psm 6 -c tessedit_char_whitelist='
                                                      f'{alphabet}')
        result += r
    
    result = "".join(filter(lambda x: x in alphabet, result.replace("\n", "").replace(" ", "").upper()))
    return result

def format_image(image):
    grayscale_resize_test_license_plate = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayscale_resize_test_license_plate,
                                0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    symbols = []
    used_pixels = set()
    height, width = image.shape[:2]
    intersected_pixels_upper_bound = width*height*0.002
    
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = thresh[y - 2:y + h + 2, x - 2:x + w + 2]
        roi = cv2.bitwise_not(roi)
        if roi is None: continue

        area = h * w
        area_ratio = area / (height * width)

        if not (0.01 < area_ratio < 0.4):
            continue

        sides_ratio = w / h

        if not (0.1 < sides_ratio < 4):
            continue

        roi_used_pixels = set((i,j) for i in range(x, x+w) for j in range(y, y + h))
        pixels_intersection_size = len(used_pixels.intersection(roi_used_pixels))

        if pixels_intersection_size > intersected_pixels_upper_bound:
            continue

        used_pixels = used_pixels.union(roi_used_pixels)
        symbols.append(roi)

    return symbols