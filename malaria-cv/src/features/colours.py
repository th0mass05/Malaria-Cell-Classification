# Colour feautures for one cell image
# Turns one RGB image into a feature vector made of an HSV colour histogram, and colour moments

import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from scipy.stats import skew

# Settings

BINS = (8, 8 , 8)
THRESHOLD = 10
HISTOGRAM_LENGTH = BINS[0] * BINS[1] * BINS[2]
MOMENTS_LENGTH = 18
FEATURE_LENGTH = HISTOGRAM_LENGTH + MOMENTS_LENGTH

# Get rid of background
# Takes an image and returns a boolean array of the shape
# True means theres a cell pixel, false means its the background

def cell_mask(image):
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = grey > THRESHOLD

    # Fill dark holes within the cell
    mask = binary_fill_holes(mask)
    return mask

# Produces a normalised 3d colour histogram 
# Takes an image and a mask and returns an array of 512 nums summing to 1
def histogram(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Convert mask as 0-255 instead of boolean
    formatted_mask = np.uint8(mask) * 255

    hist = cv2.calcHist(
        images = [hsv],
        channels = [0, 1, 2],
        mask = formatted_mask,
        histSize = list(BINS),
        ranges = [0, 180, 0, 256, 0, 256]
    )

    # flatten 
    hist = hist.flatten()

    # Divide by total so the values sum to 1
    total = hist.sum()
    if total > 0:
        hist = hist / total

    return hist

# Takes the image and mask and gives you 18 numbers, mean, stdf, and skew of each RGB AND HSV channel

def moments(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    #Combine hsv and rgb

    channels = [
        image[:, :, 0],
        image[:, :, 1],
        image[:, :, 2],
        hsv[:, :, 0],
        hsv[:, :, 1],
        hsv[:, :, 2],
    
    ]

    moments = []
    for channel in channels:
        values = channel[mask]

        # check if empty or 1 pixel
        if values.size < 2:
            moments.extend([0.0, 0.0, 0.0])
            continue

        mean = values.mean()
        std = values.std()

        # avoid undefined error if every pixel is the same

        if std > 0:
            skewness = skew(values)
        else:
            skewness = 0.0

        moments.extend([mean, std, skewness])

    moments_return = np.array(moments)

    return moments_return


def extract(image, mask = None):
    # Makes a mask if none given (tester for now)

    if mask is None:
        mask = cell_mask(image)

    hist = histogram(image, mask)
    moms = moments(image, mask)
    feature_vector = np.concatenate([hist, moms])
    return feature_vector