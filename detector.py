from detectron2.config import get_cfg
import cv2
from detectron2.engine import DefaultPredictor
import pytesseract
from detectron2.model_zoo import model_zoo
from numpy import ndarray
from typing import Dict, List, Any


class Detector:
    """
        class to get image data (bounding boxes of license plates with description)
    """
    def __init__(self):
        self.config_path = 'custom_config.yml'
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
        self.cfg.merge_from_file(self.config_path)
        self.predictor = DefaultPredictor(self.cfg)

    def get_image_data(self, image: ndarray) -> List[Dict[str, Any]]:
        """
        get image data (bounding boxes of license plates with description)
        :param image: opencv image
        :return: image data
        """
        bounding_boxes = self._find_bounding_boxes(image)
        boxes_with_text = [{'box': box, 'text': self._get_text(image, box)} for box in bounding_boxes]
        return boxes_with_text

    def _find_bounding_boxes(self, image: ndarray) -> List[List[float]]:
        """
        get bounding boxes of image
        :param image: opencv image
        :return: list with bounding boxes
        """
        grey_scaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(grey_scaled, cv2.COLOR_GRAY2RGB)
        outputs = self.predictor(gray_rgb)
        bounding_boxes = outputs['instances'].get_fields()['pred_boxes'].tensor.tolist()
        return bounding_boxes

    def _get_text(self, image: ndarray, bb: List[float]) -> str:
        """
        get test image bounding box

        :param image: opencv image
        :param bb: bounding box to get text from
        :return: text from boudning box
        """
        bb = [int(i) for i in bb]
        plate_box = image[bb[1]:bb[3], bb[0]:bb[2]]
        symbols = self._format_image(plate_box)
        result = ""
        for s in symbols:
            result += pytesseract.image_to_string(s, lang='eng',
                                                  config='--oem 3 --psm 6 -c tessedit_char_whitelist='
                                                         'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        result = result.replace("\n", "").replace(" ", "").upper()
        return result

    @staticmethod
    def _format_image(image):
        """
        format image to get text

        :param image: opencv image
        :return: processed image
        """
        grayscale_resize_test_license_plate = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gaussian_blur_license_plate = cv2.GaussianBlur(grayscale_resize_test_license_plate, (5, 5), 0)
        ret, thresh = cv2.threshold(gaussian_blur_license_plate,
                                    0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        symbols = []

        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            height, width = image.shape[:2]
            roi = thresh[y - 5:y + h + 5, x - 5:x + w + 5]
            roi = cv2.bitwise_not(roi)
            roi = cv2.medianBlur(roi, 5)
            if roi is None: continue
            area = h * w

            area_ratio = area / (height * width)
            if not (0.02 < area_ratio < 0.4):
                continue

            sides_ratio = w / h
            if not (0.25 < sides_ratio < 4):
                continue
            symbols.append(roi)

        return symbols
