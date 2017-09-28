try:
    import cv2 as cv
except ImportError:
    print('OpenCV module not installed. Please install with:\n  pip install opencv-python')


def main():
    inputImage = cv.imread('input.jpg')

    cv.imshow('image', inputImage)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
