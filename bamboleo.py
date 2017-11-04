#!/usr/bin/env python3
#
# Required Python: Python 3.5+
# Required packages: See ImportError message below

import os
import sys
import json
import argparse
import operator

# 3rd party modules
try:
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    from dxfwrite import DXFEngine as dxf, const as dxfconst
except ImportError as e:
    print(str(e) + '\nPlease install required packages with:\n  pip3 install opencv-python matplotlib numpy dxfwrite')
    sys.exit(1)


# Global settings
args = None

#
# Utils
#

# Units
def dpi2dpcm(dpi): return dpi * 0.393701
def px2cm(pixels): return pixels / dpi2dpcm(args.dpi)
def px2mm(pixels): return 10 * px2cm(pixels)
def cm2px(cm): return cm * dpi2dpcm(args.dpi)

# Plotting
figureCount = 0
class Figure:
    ''' Scope-based plotting util that will name and save figures in the output directory. '''
    def __init__(self, title='', filetype='png'):
        self.title = title
        self.filetype = filetype
    def __enter__(self):
        plt.figure()
        plt.axis('off')
        if self.title:
            plt.title(self.title, loc='left')
            self.title = self.title.replace('\n', ' ')
    def __exit__(self, type, value, traceback):
        global figureCount
        path = os.path.join(args.output, f'figure_{figureCount}.{self.filetype}')
        print(f'> Saving figure {figureCount}: {self.title}')
        plt.savefig(path, bbox_inches='tight', dpi=250)
        plt.close()
        figureCount += 1

class Contour:
    ''' Wraps a contour obtained with findContours '''
    def __init__(self, points, treeNode=None):
        self.points = points
        self.treeNode = treeNode
        # Place-holder properties. Functions will be called the first time the property function is called.
        self._area = cv2.contourArea
        self._rect = cv2.boundingRect
        self._ellipse = cv2.fitEllipse
        self._moments = cv2.moments
        self._perimeter = cv2.arcLength

    def area(self):
        if callable(self._area): self._area = self._area(self.points)
        return self._area
    def rect(self):
        ''' (x, y, width, height) '''
        if callable(self._rect): self._rect = self._rect(self.points)
        return self._rect
    def ellipse(self):
        if len(self.points) < 5: return None
        if callable(self._ellipse): self._ellipse = self._ellipse(self.points)
        return self._ellipse
    def moments(self):
        if callable(self._moments): self._moments = self._moments(self.points)
        return self._moments
    def perimeter(self):
        if callable(self._perimeter): self._perimeter = self._perimeter(self.points, True)
        return self._perimeter

    def centroid(self):
        m = self.moments()
        return (m['m10'] / m['m00'], m['m01'] / m['m00'])
    def aspectRatio(self):
        _, _, width, height = self.rect()
        return width / height
    def equivalentDiamater(self):
        return np.sqrt(4 * self.area() / np.pi)

    def __repr__(self):
        return f'[rect: {self.rect()}, area: {self.area()} px]'

def getContours(contourArray, treeArray, epsilon=None):
    '''
    Create a list of Contour from the output of cv2.findContour
    If epsilon is not None then it will be used to simplify the polygon.
    '''
    contours = [Contour(contourArray[i], treeArray[0, i, :]) for i in range(len(contourArray))]
    if epsilon:
        for c in contours:
            e = epsilon * cv2.arcLength(c.points, True)
            c.points = cv2.approxPolyDP(c.points, e, True)
    return contours

def contourMayBeTube(contour, minRectSideCm=2, maxAspectRatioError=0.3):
    '''
    Basic tube-ness test. Returns false when contour is for sure not a valid tube.
    TODO: Maybe check how well it fits an ellipse
    '''
    # Check aspect ratio
    if not ((1 - maxAspectRatioError) < contour.aspectRatio() < (1 + maxAspectRatioError)):
        return False

    _, _, width, height = contour.rect()
    return width > cm2px(minRectSideCm) and height > cm2px(minRectSideCm)

def contourMayBeHole(contour):
    '''
    Basic hole-ness test. Returns false when contour is for sure not a valid tube hole.
    Use only on contours inside a contourMayBeTube() contour.
    '''
    return contourMayBeTube(contour, minRectSideCm=1)

#
# First stage: Image segmentation to find tube contours (two approaches)
#

def findTopLevelContours1(grayscale):
    print('Finding top-level contours using flood-fill segmentation.')
    height, width = grayscale.shape[:2]

    # Apply flood-fill on the black background
    # TODO: If a fixed seed causes problem iterate with random seed positions near the corners
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(grayscale, mask, (5, 5), 255, 2, 12, flags=4 | cv2.FLOODFILL_MASK_ONLY)
    # Crop mask to image size, invert to have 1 on the tubes and 0 on the background
    mask = mask[1:height+1, 1:width+1]
    mask = 1 - mask

    # Apply strong morphological opening to remove false-positives
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
    filteredMask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Extract top-level contours and organize them in a list of Contour objects.
    _, contourArray, treeArray = cv2.findContours(filteredMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = getContours(contourArray, treeArray)

    # Make contours convex
    for c in contours:
        c.points = cv2.convexHull(c.points)

    if args.plots:
        with Figure('Background flood-fill mask'):
            plt.imshow(mask * 255, cmap='gray')
        with Figure('Filtered mask and detected tube rectangles'):
            render = np.zeros((height, width, 3), np.uint8)
            render[filteredMask == 1] = (255, 255, 255)
            for c in contours:
                x, y, w, h = c.rect()
                cv2.rectangle(render, (x, y), (x+w, y+h), (0, 255 ,0), 2)
            plt.imshow(render)

    return contours

def findTopLevelContours2(grayscale):
    print('Finding top-level contours using global-histogram segmentation.')

    # Binarize image. Image is mostly bimodal, so Otsu is a good fit.
    threshold, binarized = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)

    # Apply morphological closing (dilatation followed by erosion) to close some of the internal holes and
    # reduce the number of contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filteredMask = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Extract top-level contours and organize them in a list of Contour objects.
    _, contourArray, treeArray = cv2.findContours(filteredMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = getContours(contourArray, treeArray)

    if args.plots:
        with Figure('Luminance histogram'):
            plt.axis('on'), plt.xticks([]), plt.yticks([])
            plt.hist(grayscale.ravel(), 128)
            plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2)
        with Figure(f'Binarized image (threshold = {int(threshold)})'):
            plt.imshow(binarized, cmap='gray')
        with Figure(f'Filtered mask with possible contour rects'):
            height, width = grayscale.shape[:2]
            render = np.zeros((height, width, 3), np.uint8)
            render[filteredMask != 0] = (255, 255, 255)
            for c in contours:
                x, y, w, h = c.rect()
                cv2.rectangle(render, (x, y), (x+w, y+h), (0, 255 ,0), 2)
            plt.imshow(render)

    return contours

def extractTube(inputImage, contour, tubeId):
    def scaledPoints(contour, k):
        ''' Returns .points of a Contour scaled around its center. '''
        center = np.array(contour.centroid())
        points = k * (contour.points - center) + center
        return points.astype(np.int32, copy=False)

    def scaledRect(rect, k, width, height):
        ''' Scale rect maintaining center and clamp to (0..width-1, 0..height-1). '''
        x, y, w, h = rect
        pos, size = np.array((x, y)), np.array((w, h))
        center = pos + size / 2
        pos, size = k * (pos - center) + center, k * size
        pos = np.clip(pos, [0, 0], [width-1, height-1]).astype(int)
        size = np.clip(pos + size, [0, 0], [width-1, height-1]).astype(int) - pos
        return (pos[0], pos[1], size[0], size[1])

    print(f'* Processing tube {tubeId}...')

    # Scale the contour by 1.1x and to get a bounding rect that contains every tube pixel.
    # TODO: If the contours are noise either filter them or use ellipses for this
    inputHeight, inputWidth = inputImage.shape[:2]
    x, y, w, h = scaledRect(contour.rect(), 1.1, inputWidth, inputHeight)
    # Crop the expanded rect tube
    tubeImage = inputImage[y:y+h, x:x+w]

    # Create and empty "sure background" mask then draw concentric contours with different labels
    # Mask is full input size to avoid problems scaling the cropped contour
    mask = np.full((inputHeight, inputWidth), cv2.GC_BGD, np.uint8)
    hints = [
        (1.05, cv2.GC_PR_BGD),
        (1.00, cv2.GC_PR_FGD),
        (0.90, cv2.GC_FGD),
        (0.80, cv2.GC_PR_BGD),
        (0.25, cv2.GC_BGD)
    ]
    for scale, label in hints:
        cv2.drawContours(mask, [scaledPoints(contour, scale)], -1, label, -1)
    # Crop mask
    mask = mask[y:y+h, x:x+w]

    # Run graph-cuts
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    outputMask = mask.copy()
    cv2.grabCut(tubeImage, outputMask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # Convert graph-cuts label output to a binary mask (1 for foreground)
    outputMask = np.where((outputMask == cv2.GC_BGD) | (outputMask == cv2.GC_PR_BGD), 0, 1).astype('uint8')

    # Morphologic opening to discard small islands
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    outputMask = cv2.morphologyEx(outputMask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Extract full contour hierarchy from the output mask
    # Use polygon approximation and simplify the final contours
    _, contourArray, treeArray = cv2.findContours(outputMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    tubeContours = getContours(contourArray, treeArray, epsilon=0.0005)

    # Find the largest top-level contour. This is the outside tube contour.
    topLevelContours = [c for c in tubeContours if c.treeNode[3] == -1]
    outsideContour = max(topLevelContours, key=operator.methodcaller('area'))

    # Find the largest contour which is contained in the top-level contour. This is the inside tube contour.
    secondLevelContours = [c for c in tubeContours if c.treeNode[3] == topLevelContours.index(outsideContour)]
    insideContour = max(secondLevelContours, key=operator.methodcaller('area'))

    print(f'* Found {len(topLevelContours)} top-level contours, {len(secondLevelContours)} sub-contours.')

    if args.plots:
        with Figure(f'Tube {tubeId}: Graph-cuts input labels'):
            renderMask = np.zeros((h, w, 3), np.uint8)
            renderMask[mask == cv2.GC_FGD] = (0, 200, 0)
            renderMask[mask == cv2.GC_BGD] = (255, 0, 0)
            render = cv2.addWeighted(tubeImage, 0.7, renderMask, 0.3, 0)
            plt.imshow(render)
        with Figure(f'Tube {tubeId}: Graph-cuts filtered output'):
            renderMask = np.zeros((h, w, 3), np.uint8)
            renderMask[outputMask == 1] = (0, 200, 0)
            renderMask[outputMask == 0] = (255, 0, 0)
            render = cv2.addWeighted(tubeImage, 0.7, renderMask, 0.3, 0)
            plt.imshow(render)
    # Plot final contour
    odCm = px2cm(outsideContour.equivalentDiamater())
    idCm = px2cm(insideContour.equivalentDiamater())
    solidPerc = (outsideContour.area() - insideContour.area()) / outsideContour.area() * 100
    with Figure(f'Tube {tubeId + 1}:\nOD = {odCm:.1f} cm, ID = {idCm:.1f} cm, {int(solidPerc)}% solid.'):
        render = tubeImage.copy()
        cv2.drawContours(render, [outsideContour.points], -1, (255, 0, 255), 2)
        cv2.drawContours(render, [insideContour.points], -1, (255, 255, 0), 2)
        plt.imshow(render)

    return (outsideContour, insideContour, tubeImage)

def main():
    # Load image
    inputImageBGR = cv2.imread(args.inputPath)
    if inputImageBGR is None:
        print(f'Failed to open image file "{args.inputPath}".')
        return
    inputHeight, inputWidth = inputImageBGR.shape[:2]
    print(f'Loaded "{args.inputPath}": {inputWidth}x{inputHeight} px at {args.dpi} DPI is {px2cm(inputWidth):.1f}x{px2cm(inputHeight):.1f} cm.')

    # Convert BGR to RGB
    inputImage = cv2.cvtColor(inputImageBGR, cv2.COLOR_BGR2RGB)
    # Convert to grayscale
    grayscale = cv2.cvtColor(inputImageBGR, cv2.COLOR_RGB2GRAY)

    if args.plots:
        with Figure('Input image'):
            plt.imshow(inputImage)

        with Figure('Luminance (false color)'):
            plt.imshow(grayscale, cmap='jet')
            plt.colorbar(orientation='horizontal')

    # Get candidate tube contours
    topLevelContours = findTopLevelContours1(grayscale)

    # Discard contours that fail a basic tube-ness test.
    tubesContours = [c for c in topLevelContours if contourMayBeTube(c)]

    if not tubesContours:
        print('No tubes found!')
        return

    print(f'Found {len(tubesContours)} tubes:')
    for i, tube in enumerate(tubesContours):
        height, width = tube.rect()[2:]
        print(f'  Tube {i+1}: Bounding rectangle size {px2cm(width)*10:.0f}x{px2cm(height)*10:.0f} mm.')
    if len(tubesContours) != len(topLevelContours):
        print(f'Discarded {len(topLevelContours)-len(tubesContours)} contours deemed too small.')

    for tubeId, contour in enumerate(tubesContours):
        outsideContour, insideContour, tubeRectImage = extractTube(inputImage, contour, tubeId + 1)

        # Save contour JSON, all values are relative to the tube rectangle and not the original image coords
        x0, y0, width, height = outsideContour.rect()
        origin = np.array([x0, y0])
        size = np.array([width, height])
        outsidePointsMM = px2mm(np.squeeze(outsideContour.points) - origin - size/2.0).tolist()
        insidePointsMM = px2mm(np.squeeze(insideContour.points) - origin - size/2.0).tolist()
        jsonOut = {
            'tubeId' : tubeId,
            'units' : 'mm',
            'width' : px2mm(width),
            'height' : px2mm(height),
            'outsideContour' : outsidePointsMM,
            'insideContour' : insidePointsMM
        }
        jsonPath = os.path.join(args.output, f'tube_{tubeId}.json')
        open(jsonPath, 'w').write(json.dumps(jsonOut, indent=4))

        # Save contour DXF
        dxfPath = os.path.join(args.output, f'tube_{tubeId}.dxf')
        drawing = dxf.drawing(dxfPath)
        drawing.header['$INSUNITS'] = 4 # Units are in mm
        drawing.header['$MEASUREMENT'] = 1 # Do measurements in metric
        drawing.add(dxf.polyline(points=outsidePointsMM, flags=dxfconst.POLYLINE_CLOSED))
        drawing.add(dxf.polyline(points=insidePointsMM, flags=dxfconst.POLYLINE_CLOSED))
        drawing.save()

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('inputPath', help='image to process', metavar='image.jpg')
    parser.add_argument('dpi', help='image dots-per-inch', metavar='dpi', type=int)
    parser.add_argument('--output', help='output directory', metavar=f'output{os.path.sep}', default=None)
    parser.add_argument('--plots', help='generate intermediate plots', action='store_true')
    args = parser.parse_args()

    # Default output dir is the same as the file without the extension
    args.output = args.output or os.path.splitext(args.inputPath)[0]
    # Get absolute path out output dir, create if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print(f'Output directory is "{args.output}".')

    main()
