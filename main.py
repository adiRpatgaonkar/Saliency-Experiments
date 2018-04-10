"""
Saliency based maps for images as ground truths
"""
from __future__ import print_function

from collections import OrderedDict as OD
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import numpy as np
import MR
import pyimgsaliency as psal


filename = "COCO_train2014_000000349201.jpg"

# Load image from disk
ground_truth = OD()

mr = MR.MR_saliency()

image = cv2.imread(filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(121)
plt.imshow(image)
#plt.show()

# ++++ Saliency for original image ++++ #
# manifold_image = mr.saliency(image).astype('uint8')
# binary_sal_mfd = psal.binarise_saliency_map(manifold_image, 
#                                             method='adaptive', 
#                                             threshold=0.5,
#                                             adaptive_factor=0.98)

# mbd = psal.get_saliency_mbd(filename).astype('uint8')
# binary_sal_mbd = psal.binarise_saliency_map(mbd, 
#                                             method='adaptive', 
#                                             threshold=0.1,
#                                             adaptive_factor=0.95)

# rbd = psal.get_saliency_rbd(filename).astype('uint8')
# binary_sal_rbd = psal.binarise_saliency_map(rbd, 
#                                             method='adaptive', 
#                                             threshold=0.1,
#                                             adaptive_factor=1.5)

# ftu = psal.get_saliency_ft(filename).astype('uint8')
# binary_sal_ftu = psal.binarise_saliency_map(ftu, 
#                                             method='adaptive', 
#                                             threshold=0.01,
#                                             adaptive_factor=1)

# ++++ ROI ++++ #
# FORMAT: (x, y, width, height)
imgs_bb_data = [(240, 213, 85, 60), (300, 300, 100, 100)]
roi_file = "test.jpg"
for bb_data in imgs_bb_data[:1]:
    # Extract ROI
    print("To extract:\n{}".format(bb_data))
    x, y = bb_data[0:2]
    print(x, y)
    w = x + bb_data[2]
    h = y + bb_data[3]
    ground_truth[bb_data] = image[y:h, x:w, :]
    print("Extracted:\n{}".format(ground_truth[bb_data].shape))
    plt.subplot(122)
    plt.imshow(ground_truth[bb_data])

plt.show()

roi_image = Image.fromarray(ground_truth[bb_data])
roi_image.save(roi_file)

mbd_roi = psal.get_saliency_mbd(roi_file).astype('uint8')
binary_sal_mbd_roi = psal.binarise_saliency_map(mbd_roi, 
                                            method='adaptive', 
                                            threshold=0.1,
                                            adaptive_factor=0.95)
plt.imshow(binary_sal_mbd_roi, cmap='Greys',  interpolation='nearest')
plt.title('MBD')
plt.show()

rbd_roi = psal.get_saliency_rbd(roi_file).astype('uint8')
binary_sal_rbd_roi = psal.binarise_saliency_map(rbd_roi, 
                                             method='adaptive', 
                                             threshold=0.1,
                                             adaptive_factor=0.268)

plt.imshow(binary_sal_rbd_roi, cmap='Greys',  interpolation='nearest')
plt.title('Robust background detection')
plt.show()

ftu_roi = psal.get_saliency_ft(roi_file).astype('uint8')
binary_sal_ftu_roi = psal.binarise_saliency_map(ftu_roi, 
                                             method='adaptive', 
                                             threshold=0.01,
                                             adaptive_factor=0.5)
plt.imshow(binary_sal_ftu_roi, cmap='Greys',  interpolation='nearest')
plt.title('Frequency tuning')
plt.show()