# This script requires opencv-python and matplotlib. They can be installed with:
#   pip install opencv-python matplotlib

import sys
import argparse

# Import and check 3rd party modules
try:
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
except ImportError:
    print('OpenCV module not installed. Please install with:\n  pip install opencv-python')
    sys.exit(1)

def main(inputPath):
    inputImage = cv2.imread(inputPath)
    if inputImage is None:
        print(f'Failed to open image file "{inputPath}".')
        return

    grayscale = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Image is mostly bimodal, so Otsu binarization is a good fit
    threshold, binarized = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow(f'Binarized image, threshold = {threshold}', binarized)

    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel, iterations = 2)

    cv2.imshow(f'Opened', opening)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputPath', help='image to process', metavar='image.jpg')
    args = parser.parse_args()

    main(args.inputPath)
