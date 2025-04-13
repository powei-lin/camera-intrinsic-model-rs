import cv2
import numpy as np
from time import perf_counter


def main():
    img = (255 * np.random.random((512, 512))).astype(np.uint8)
    mapx = (511 * np.random.random((1024, 1024))).astype(np.float32)
    mapy = (511 * np.random.random((1024, 1024))).astype(np.float32)
    # mapx, mapy = cv2.convertMaps(mapx, mapy, cv2.CV_16SC2)
    tt = 1000
    t0 = perf_counter()
    for i in range(tt):
        cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    t1 = perf_counter()
    print(f"{(t1 - t0) / tt}")


if __name__ == "__main__":
    main()
