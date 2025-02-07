import cv2
import os
import requests
import json
from base64 import b64encode
import time

YC_API_KEY = os.getenv("YC_API_KEY")
assert YC_API_KEY
print(f"api_key: {YC_API_KEY[:8]}")

def YC_OCR_makeImageData(imgpath):
  with open(imgpath, 'rb') as fid:
      file_content = fid.read()
  return b64encode(file_content).decode('utf-8')


def ocr_detection_yc2(img):
    enc_img = b64encode(img).decode('utf-8')
    data = {"mimeType": "JPEG",
            "languageCodes": ["*"],
            "content": enc_img}

    url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Api-Key {:s}".format(YC_API_KEY),
        "x-data-logging-enabled": "true"
    }
    start = time.monotonic()
    w = requests.post(url=url, headers=headers, data=json.dumps(data))
    w.raise_for_status()
    print('*** Text Detection Time Taken:%.3fs ***' % (time.monotonic() - start))
    result = []
    for block in w.json()["result"]["textAnnotation"]["blocks"]:
        #print(json.dumps(block, indent=2))
        for line in block["lines"]:
            for w in line.get("words", []):
                verts =[{"x": int(v["x"]), "y": int(v["y"])} for v in w["boundingBox"]["vertices"]]                
                result.append({
                    "description": w["text"],
                    "boundingPoly": {
                        "vertices": verts
                    }
                })

    return result


def ocr_detection_yc(imgpath):
    data = {"mimeType": "JPEG",
            "languageCodes": ["*"],
            "content": YC_OCR_makeImageData(imgpath)}

    url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Api-Key {:s}".format(YC_API_KEY),
        "x-data-logging-enabled": "true"
    }
    start = time.monotonic()
    w = requests.post(url=url, headers=headers, data=json.dumps(data))
    w.raise_for_status()
    print('*** Text Detection Time Taken:%.3fs ***' % (time.monotonic() - start))
    result = []
    for block in w.json()["result"]["textAnnotation"]["blocks"]:
        #print(json.dumps(block, indent=2))
        for line in block["lines"]:
            for w in line.get("words", []):
                verts =[{"x": int(v["x"]), "y": int(v["y"])} for v in w["boundingBox"]["vertices"]]                
                result.append({
                    "description": w["text"],
                    "boundingPoly": {
                        "vertices": verts
                    }
                })

    return result


if __name__ == "__main__":
    r = ocr_detection_yc("C:\\Users\\thetweak\\source\\UIED\\data\\input\\497.jpg")
    print(r)