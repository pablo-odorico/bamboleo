#!/usr/bin/env python3
#
# Required Python: Python 3.5+
# Required packages: opencv-python matplotlib numpy

import os
import sys
import argparse

# 3rd party modules
try:
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
except ImportError as e:
    print(str(e) + '\nPlease install required packages with:\n  pip3 install opencv-python matplotlib numpy')
    sys.exit(1)

# Global settings
args = None
enablePlots = True

#
# Utils
#

# Units
def dpi2dpcm(dpi): return dpi * 0.393701
def px2cm(pixels): return pixels / dpi2dpcm(args.dpi)
def cm2px(cm): return cm * dpi2dpcm(args.dpi)

figureCount = 0
class Figure():
    ''' Scope-based plotting util that will name and save figures in the output directory. '''
    def __init__(self, title=''):
        self.title = title
    def __enter__(self):
        plt.figure()
        plt.axis('off')
        if self.title: plt.title(self.title, loc='left')
    def __exit__(self, type, value, traceback):
        global figureCount
        path = os.path.join(args.output, f'figure_{figureCount}.png')
        print(f'> Saving figure {figureCount}: {self.title}')
        plt.savefig(path, bbox_inches='tight', dpi=250)
        plt.close()
        figureCount += 1

class Contour:
    def __init__(self, contour, treeNode):
        self.contour = contour
        self.treeNode = treeNode
        self._area = None
        self._rect = None

    def area(self):
        if self._area is None: self._area = cv2.contourArea(self.contour)
        return self._area
    def rect(self):
        ''' Returns (x, y, w, h) '''
        if self._rect is None: self._rect = cv2.boundingRect(self.contour)
        return self._rect

    def __repr__(self):
        return f'[rect: {self.rect()}, area: {self.area()} px]'

def Contours(contourArray, treeArray):
    ''' Create a list of Contour from the output of cv2.findContour '''
    return [Contour(contourArray[i], treeArray[0, i, :]) for i in range(len(contourArray))]

#
# First stage: Image segmentation and finding cut contours (two approaches)
#

def findTopLevelContours1(grayscale):
    print('Finding top-level contours using flood-fill segmentation.')
    height, width = grayscale.shape[:2]

    # Apply flood-fill on the black background
    # TODO: If a fixed seed causes problem iterate with random seed positions near the corners
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(grayscale, mask, (5, 5), 255, 1, 12, flags=4 | cv2.FLOODFILL_MASK_ONLY)
    # Crop mask to image size, invert to have 1 on the cuts and 0 on the background
    mask = mask[1:height+1, 1:width+1]
    mask = 1 - mask

    # Apply strong morphological opening to remove false-positives
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    filteredMask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Extract top-level contours and organize them in a list of Contour objects.
    _, contourArray, treeArray = cv2.findContours(filteredMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = Contours(contourArray, treeArray)

    if enablePlots:
        with Figure('Background flood-fill mask'):
            plt.imshow(mask * 255, cmap='gray')
        with Figure('Filtered background mask'):
            plt.imshow(filteredMask * 255, cmap='gray')

    return contours

def findTopLevelContours2(grayscale):
    print('Finding top-level contours using global-histogram segmentation.')

    # Binarize image. Image is mostly bimodal, so Otsu is a good fit.
    threshold, binarized = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)

    # Apply morphological closing (dilatation followed by erosion) to close some of the internal holes and
    # reduce the number of contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Extract top-level contours and organize them in a list of Contour objects.
    _, contourArray, treeArray = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = Contours(contourArray, treeArray)

    if enablePlots:
        with Figure('Luminance histogram'):
            plt.axis('on'), plt.xticks([]), plt.yticks([])
            plt.hist(grayscale.ravel(), 128)
            plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2)
        with Figure(f'Binarized image (threshold = {int(threshold)})'):
            plt.imshow(binarized, cmap='gray')
        with Figure(f'Filtered mask'):
            plt.imshow(closed, cmap='gray')

    return contours

def main():
    # Load image
    print(f'Loading "{args.inputPath}".')
    inputImageBGR = cv2.imread(args.inputPath)
    if inputImageBGR is None:
        print(f'Failed to open image file "{args.inputPath}".')
        return

    # Convert BGR to RGB
    inputImage = cv2.cvtColor(inputImageBGR, cv2.COLOR_BGR2RGB)
    # Convert to grayscale
    grayscale = cv2.cvtColor(inputImageBGR, cv2.COLOR_RGB2GRAY)

    if enablePlots:
        with Figure('Input image'):
            plt.imshow(inputImage)

        with Figure('Luminance (false color)'):
            plt.imshow(grayscale, cmap='jet')
            plt.colorbar(orientation='horizontal')

    # Get candidate cut contours
    topLevelContours = findTopLevelContours1(grayscale)

    # Discard contours that are too small: both sides of the bounding rectangle have to be above a minRectSide
    minRectSide = cm2px(2.0)
    cutsContours = [c for c in topLevelContours if c.rect()[2] > minRectSide and c.rect()[3] > minRectSide]

    if not cutsContours:
        print('No cuts found!')
        return

    print(f'Found {len(cutsContours)} cuts:')
    for i, cut in enumerate(cutsContours):
        print(f'  Cut {i+1}: Area {cut.area()} px, bounding rectangle size {cut.rect()[2:]} px')
    if len(cutsContours) != len(topLevelContours):
        print(f'Discarded {len(topLevelContours)-len(cutsContours)} contours deemed too small.')

    if enablePlots:
        with Figure('Detected cuts'):
            render = inputImage.copy()
            for c in topLevelContours:
                if len(c.contour) < 5: continue
                color = (255, 0, 255) if c in cutsContours else (0, 0, 255)
                ellipse = cv2.fitEllipse(c.contour)
                cv2.ellipse(render, ellipse, color, 2)
            for c in cutsContours:
                x, y, w, h = c.rect()
                cv2.rectangle(render, (x, y), (x+w, y+h), (0, 255, 0), 2)
            plt.imshow(render)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('inputPath', help='image to process', metavar='image.jpg')
    parser.add_argument('dpi', help='image dots-per-inch', metavar='dpi', type=int)
    parser.add_argument('--output', help='output directory', metavar='output/', default='output')
    args = parser.parse_args()

    # Get absolute path out output dir, create if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f'Created output directory "{args.output}".')

    main()


'''
Watershed sample code

    opening = cv2.morphologyEx(binarized, cv2.MORPH_OPEN,kernel, iterations=2)
# sure background area
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
sure_bg = cv2.dilate(closed, kernel, iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(inputImage, markers)
inputImage[markers == -1] = [255,0,0]
#cv2.imshow('1opening',opening)
cv2.imshow('2sure_bg',sure_bg)
#cv2.imshow('3dist_transform', dist_transform / dist_transform.max() * 255)
cv2.imshow('4sure_fg',sure_fg)
#cv2.imshow('4unknown',unknown)
cv2.imshow('markers',inputImage)
cv2.waitKey()
'''
