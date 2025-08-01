import cv2
import os

data_root = './bad_frames'
os.makedirs(data_root, exist_ok=True)

def get_v_data(v: cv2.VideoCapture):
    return {
        "fps": round(v.get(cv2.CAP_PROP_FPS)),
        "total_f": int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    }

def extract_data(v: cv2.VideoCapture):
    serial = 0
    while True:
        ret, frame = v.read()  # sequential read is fast
        if not ret:
            break
        file_name = f'{data_root}/{serial}.jpg'
        cv2.imwrite(file_name, frame)
        serial += 1

def extract_video():
    v = cv2.VideoCapture("./bad_apple.mp4")
    config = get_v_data(v)
    print(config)
    extract_data(v)
    v.release()
    cv2.destroyAllWindows()

extract_video()
