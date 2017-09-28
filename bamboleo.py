# This script requires opencv-python and matplotlib. They can be installed with:
#   pip install opencv-python matplotlib

import os
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

outputDir = None

# Plotting utils
figureCount = 0
class Figure():
    def __init__(self, title=''):
        self.title = title
    def __enter__(self):
        plt.figure()
        if self.title: plt.title(self.title, loc='left')
    def __exit__(self, type, value, traceback):
        global outputDir, figureCount
        path = os.path.join(outputDir, f'step_{figureCount}.png')
        print(f'Saving "{path}".')
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
        figureCount += 1

def main(inputPath):
    # Load image
    inputImage = cv2.imread(inputPath)
    if inputImage is None:
        print(f'Failed to open image file "{inputPath}".')
        return

    # Convert BGR to RGB
    inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
    # Convert to grayscale
    grayscale = cv2.cvtColor(inputImage, cv2.COLOR_RGB2GRAY)

    # Binarize image. Image is mostly bimodal, so Otsu is a good fit.
    threshold, binarized = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)

    # Apply morphological closing (dilatation followed by erosion) to close some of the internal holes and
    # reduce the number of contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Detect contours and extract a hierarchy. Contour approximation is good enough to reduce polygon complexity.
    closed, contourArray, treeArray = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    class Contour:
        def __init__(self, contour, treeNode):
            self.contour = contour
            self.treeNode = treeNode
            self.area = cv2.contourArea(contour)

        def __repr__(self):
            return f'[area: {self.area}, vertices: {len(self.contour)}]'

    # Re-arrange the data in a single list. This list can then be filtered and split into categories.
    contours = [Contour(contourArray[i], treeArray[0, i, :]) for i in range(len(contourArray))]

    # Discard contours that are too small
    # TODO: Use real area, not pixels (will need DPI value)
    minArea = 50 * 50
    tooSmallContours = [c for c in contours if c.area < minArea]
    contours = [c for c in contours if c not in tooSmallContours]

    # Separate remaining contours in top-level and child ones
    parentContours = [c for c in contours if c.treeNode[3] == -1]
    childContours = [c for c in contours if c not in parentContours]

    # Plotting
    savePlots = True
    if not savePlots:
        return

    with Figure('Input Image'):
        plt.imshow(inputImage)
        plt.axis('off')

    with Figure('Luminance (in false color)'):
        plt.imshow(grayscale, cmap='jet')
        plt.axis('off')
        plt.colorbar(orientation='horizontal')

    with Figure('Luminance Histogram'):
        plt.hist(grayscale.ravel(), 128)
        plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2)
        plt.xticks([])
        plt.yticks([])

    with Figure(f'Binarized Image (threshold = {int(threshold)})'):
        plt.imshow(binarized, cmap='gray')
        plt.axis('off')

    with Figure(f'Morphological Closing'):
        plt.imshow(closed, cmap='gray')
        plt.axis('off')

    with Figure(f'Filtered Contours'):
        render = inputImage.copy()
        cv2.drawContours(render, [c.contour for c in tooSmallContours], -1, (255, 0, 255), 3)
        cv2.drawContours(render, [c.contour for c in parentContours], -1, (0, 255, 0), 3)
        cv2.drawContours(render, [c.contour for c in childContours], -1, (0, 255, 255), 3)
        plt.imshow(render)
        plt.axis('off')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputPath', help='image to process', metavar='image.jpg')
    parser.add_argument('--output', help='output directory', metavar='output/', default='output')
    args = parser.parse_args()

    # Get absolute path out output dir, create if it doesn't exist
    outputDir = args.output
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        print(f'Created output directory "{outputDir}".')

    main(args.inputPath)
