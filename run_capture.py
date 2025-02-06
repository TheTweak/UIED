from os.path import join as pjoin
import cv2
import numpy as np
import mss
import time


def resize_height_by_longest_edge2(org, resize_length=800):
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


def color_tips():
    board = np.zeros((200, 200, 3), dtype=np.uint8)
    board[:50, :, :] = (0, 0, 255)
    board[50:100, :, :] = (0, 255, 0)
    board[100:150, :, :] = (255, 0, 255)
    board[150:200, :, :] = (0, 255, 255)
    cv2.putText(board, 'Text', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, 'Non-text Compo', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, "Compo's Text Content", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, "Block", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow('colors', board)


if __name__ == '__main__':
    color_tips()

    key_params = {'min-grad':10, 'ffl-block':5, 'min-ele-area':50,
                  'merge-contained-ele':True, 'merge-line-to-paragraph':False, 'remove-bar':True}

    with mss.mss() as sc:
        monitor_number = 1
        mont = sc.monitors[monitor_number]
        mont["mon"] = monitor_number    
        while True:
            lt = time.monotonic()
            img = sc.grab(mont)
            npimg = np.array(img)

            resized_height = resize_height_by_longest_edge2(npimg, resize_length=800)
            compression_quality = 90
            _, encoded_image = cv2.imencode('.jpg', npimg, [cv2.IMWRITE_JPEG_QUALITY, compression_quality])
            compressed_bytes = np.array(encoded_image).tobytes()
                
            import detect_text.text_detection as text
            ocr = text.text_detection2(compressed_bytes, npimg.shape)

            import detect_compo.ip_region_proposal as ip
            classifier = None
            uicompos = ip.compo_detection2(npimg, key_params, resize_by_height=resized_height)

            import detect_merge.merge as merge
            fimage = merge.merge2(npimg, uicompos, ocr, is_remove_bar=key_params['remove-bar'], is_paragraph=key_params['merge-line-to-paragraph'])

            tp = time.monotonic() - lt
            if tp > 0:
                print(f"fps: {1/tp}")
            
            cv2.imshow("test", cv2.resize(fimage, (mont['width'], mont['height'])))
            if cv2.waitKey(2000) > 0:
                cv2.destroyAllWindows()
                exit(0)
