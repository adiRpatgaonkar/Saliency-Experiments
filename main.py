"""
Saliency based maps for images as ground truths
"""
from __future__ import print_function

# <Distribution> packages
from collections import OrderedDict as OD
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.nan)

# Custom packages
import MR
import pyimgsaliency as psal

my_dpi = 1200

def display_saliency(filename, showfig=True):
    # mfd = psal.get_saliency_mbd(filename).astype('uint8')
    # print("MFD Saliency map:", mfd)
    # mfd = mfd / 255.0
    # print("MFD Saliency map UN:", mfd)
    # plt.subplot(121)
    # plt.imshow(mfd)
    # plt.show()
    # binary_sal_mfd = psal.binarise_saliency_map(mfd,
    #                                             method = 'adaptive',
    #                                             threshold=0.1,
    #                                             adaptive_factor=1.5)
    # print("MFD Saliency map binary:", binary_sal_mfd)
    # plt.subplot(122)
    # plt.imshow(binary_sal_mfd, cmap='Greys', interpolation='nearest')
    # plt.title('Manifold Ranking')
    # if showfig:
    #     plt.show()
    # else:
    #     plt.savefig("Manifold-Ranking.png", dpi=my_dpi)

    mbd = psal.get_saliency_mbd(filename).astype('uint8')
    plt.subplot(121)
    plt.imshow(mbd, cmap=cm.gray, interpolation=None)
    ##plt.show()
    binary_sal_mbd = psal.binarise_saliency_map(mbd, 
                                                method='fixed', 
                                                threshold=0.2)
    plt.subplot(122)
    plt.imshow(binary_sal_mbd, cmap=cm.gray, interpolation=None)
    plt.title('MBD')
    if showfig:
        plt.show()
    else:
        plt.savefig("MBD.png", dpi=my_dpi)


    rbd = psal.get_saliency_rbd(filename).astype('uint8')
    # print("Unnormalized:\n", rbd) # Unnormalized saliency map
    plt.subplot(121)
    plt.imshow(rbd, cmap=cm.gray, interpolation=None)
    ##plt.show()
    binary_sal_rbd = psal.binarise_saliency_map(rbd, 
                                                method='fixed', 
                                                threshold=0.815)
    # print(binary_sal_rbd)
    plt.subplot(122)
    plt.imshow(binary_sal_rbd, cmap=cm.gray, interpolation=None)
    plt.title('Robust background detection')
    if showfig:
        plt.show()
    else:
        plt.savefig("RobustBackgroundDetection.png", dpi=my_dpi)

    # ftu = psal.get_saliency_ft(filename).astype('uint8')
    # plt.subplot(121)
    # plt.imshow(ftu)
    # ##plt.show()
    # binary_sal_ftu = psal.binarise_saliency_map(ftu, 
    #                                             method='adaptive',
    #                                             threshold=0.01,
    #                                             adaptive_factor=0.5)
    # plt.subplot(122)
    # plt.imshow(binary_sal_ftu, cmap='Greys', interpolation='nearest')
    # plt.title('Frequency tuning')
    # if showfig:
    #     plt.show()
    # else:
    #     plt.savefig("FrequencyTuning.png", dpi=my_dpi)


filename = "COCO_train2014_000000349201.jpg"

# Load image from disk
ground_truth = OD()

image = cv2.imread(filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#plt.imshow(image)
#plt.show()
#display_saliency(filename)

# ++++ Saliency for ROIs ++++ #

# BB DATA FORMAT: (x, y, width, height)
imgs_bb_data = [[240, 213, 85, 60], [440, 305, 66, 86]]

roi_file = "test.jpg"
print("\nROI: {}".format(roi_file))
for bb_data in imgs_bb_data[:1]:
    # Extract ROI
    print("To extract:\n{}".format(bb_data))
    x, y = bb_data[0:2]
    w = x + bb_data[2]
    h = y + bb_data[3]
    bb_key = tuple(bb_data)
    ground_truth[bb_key] = image[y:h, x:w, :]
    print("Extracted:\n{}".format(ground_truth[bb_key].shape))
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(ground_truth[bb_key])
    plt.show()
    roi_image = Image.fromarray(ground_truth[bb_key])
    roi_image.save(roi_file)
    display_saliency(roi_file)

# ++++ Modify ROI ++++ #
roi_file_mod = "test-1.jpg"
mod_factor = 20.0
print("\nROI MODIFIED: {}".format(roi_file_mod))
for bb_data in imgs_bb_data:
    bb_data[0] -= int((mod_factor/100) * bb_data[0])
    bb_data[1] -= int((mod_factor/100) * bb_data[1])
    bb_data[2] += int((mod_factor/100) * 6 * bb_data[2])
    bb_data[3] += int((mod_factor/100) * 6 * bb_data[3])

for bb_data in imgs_bb_data[:1]:
    # Extract modified ROI
    print("To extract:\n{}".format(bb_data))
    x, y = bb_data[0:2]
    w = x + bb_data[2]
    h = y + bb_data[3]
    bb_key = tuple(bb_data)
    ground_truth[bb_key] = image[y:h, x:w, :]
    print("Extracted:\n{}".format(ground_truth[bb_key].shape))
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(ground_truth[bb_key])
    plt.show()
    roi_image_mod = Image.fromarray(ground_truth[bb_key])
    roi_image_mod.save(roi_file_mod)
    display_saliency(roi_file_mod)  