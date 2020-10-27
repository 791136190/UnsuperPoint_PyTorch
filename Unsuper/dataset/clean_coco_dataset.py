
from pathlib import Path
import cv2
import os

def clean_data_set():
    base_path = Path('../../Data/', 'COCO/val2017/')
    image_paths = list(base_path.iterdir())
    for p in image_paths:
        p = str(p)
        img = cv2.imread(p)
        if img is None:
            print(p)
            os.remove(p)
            print('eeeeeeeee!!!!')

if __name__ == '__main__':
    clean_data_set()
    print('end process!!!')