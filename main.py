"""
Saliency based maps for images as ground truths
"""
from __future__ import print_function

# <Distribution> packages
from subprocess import call
import sys
import os
from collections import OrderedDict as OD

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from skimage import feature
import cv2
from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.nan)

# Custom packages
import MR
import pyimgsaliency as psal

my_dpi = 1200

def display_saliency(filename, sal_types, showfig=True):

    types = sal_types.keys()
    print(types)
    if (not 'mfd' in types and not 'mbd' in types and
        not 'rbd' in types and not 'ft' in types):
        print("Specify a proper saliency technique.")

    if 'mfd' in types:
        #thresh = 0.5
        mfd = psal.get_saliency_mbd(filename).astype('uint8')
        plt.subplot(121)
        plt.imshow(mfd, cmap=cm.gray, interpolation=None)
        plt.title('MFD Saliency Map')
        ##plt.show()
        binary_sal_mfd = psal.binarise_saliency_map(mfd,
                                                    method = 'fixed',
                                                    threshold=sal_types['mfd'])
        plt.subplot(122)
        plt.imshow(binary_sal_mfd, cmap=cm.gray, interpolation=None)
        plt.title('MFD Binary Mask: ' + str(sal_types['mfd']))
        plt.show()
        # if showfig:
        #     plt.show()
        # else:
        #     plt.savefig("Manifold-Ranking.png", dpi=my_dpi)

    if 'mbd' in types:
        #thresh = 0.05
        mbd = psal.get_saliency_mbd(filename).astype('uint8')
        plt.subplot(121)
        plt.imshow(mbd, cmap=cm.gray, interpolation=None)
        plt.title('MBD Saliency Map')
        ##plt.show()
        binary_sal_mbd = psal.binarise_saliency_map(mbd, 
                                                    method='fixed', 
                                                    threshold=sal_types['mbd'])
        plt.subplot(122)
        plt.imshow(binary_sal_mbd, cmap=cm.gray, interpolation=None)
        plt.title('MBD Binary Mask: ' + str(sal_types['mbd']))
        plt.show()
        # if showfig:
        #     plt.show()
        # else:
        #     plt.savefig("MBD.png", dpi=my_dpi)

    if 'rbd' in types:
        #thresh = 0.7
        rbd = psal.get_saliency_rbd(filename).astype('uint8')
        # print("Unnormalized:\n", rbd) # Unnormalized saliency map
        #rbd = feature.canny(rbd, sigma=3)
        #plt.imshow(rbd, cmap=cm.gray, interpolation=None)
        #plt.show()
        plt.subplot(121)
        plt.imshow(rbd, cmap=cm.gray, interpolation=None)
        plt.title('RBD Saliency Map')
        ##plt.show()
        binary_sal_rbd = psal.binarise_saliency_map(rbd, 
                                                    method='fixed', 
                                                    threshold=sal_types['rbd'])
        plt.subplot(122)
        plt.imshow(binary_sal_rbd, cmap=cm.gray, interpolation=None)
        plt.title('RBD Binary Mask: ' + str(sal_types['rbd']))
        plt.show()
        # if showfig:
        #     plt.show()
        # else:
        #     plt.savefig("RobustBackgroundDetection.png", dpi=my_dpi)

    if 'ft' in types:
        #thresh = 0.5
        ftu = psal.get_saliency_ft(filename).astype('uint8')
        plt.subplot(121)
        plt.imshow(ftu)
        plt.title('FT Saliency Map')
        ##plt.show()
        binary_sal_ftu = psal.binarise_saliency_map(ftu, 
                                                    method='fixed',
                                                    threshold=sal_types['ft'])
        plt.subplot(122)
        plt.imshow(binary_sal_ftu, cmap=cm.gray, interpolation=None)
        plt.title('FT Binary Mask: ' + str(sal_types['ft']))
        plt.show()
        # if showfig:
        #     plt.show()
        # else:
        #     plt.savefig("FrequencyTuning.png", dpi=my_dpi)


def extract_roi(image, bbox):
    print("\nEXTRACTING RoI ... ", end="")
    print("DONE.")
    x, y = bbox[0:2]
    w, h = x + bbox[2], y + bbox[3]
    roi = image[y:h, x:w, :]
    print("Extracted: {}".format(roi.shape))
    return roi

def modify_roi(bbox, factor=10.0):
    if not isinstance(factor, float):
        factor = float(factor)
    print("\nEXPANDING RoI BY {} % ... ".format(factor), end="")
    new_bbox = [0.0] * 4
    new_width = bbox[2] + int((factor/100) * bbox[2])
    new_height = bbox[3] + int((factor/100) * bbox[3])
    new_x = (new_width - bbox[2]) / 2 # x-displacement
    new_y = (new_height - bbox[3]) / 2 # y-displacement
    bbox[0] -= new_x # new x
    bbox[1] -= new_y # new y
    bbox[2:] = new_width, new_height
    new_bbox = bbox
    print("DONE.")
    return new_bbox

def draw_bbox(image, bbox, title='RoI', color=(0, 255, 0)):
    x, y = bbox[0:2]
    w, h = x + bbox[2], y + bbox[3]
    temp = np.copy(image)
    plt.imshow(cv2.rectangle(temp, (x, y), (w, h), color, 2))
    plt.title(title)
    return temp

def save_image2disk(image, filename):
    image = Image.fromarray(image)
    image.save(filename)

# Original image files
filenames = ("COCO_train2014_000000349201.jpg", 
             "COCO_train2014_000000349267.jpg")
output_dir = "outputs/"
if not os.path.exists(output_dir):
    call("mkdir " + output_dir[:-1])

mod_factor = 50.0
# For GTs to be created
ground_truth = OD()

# BB DATA
# BB DATA FORMAT: (x, y, width, height)
imgs_bb_data = OD()
imgs_bb_data[filenames[0]] = OD()
imgs_bb_data[filenames[0]]['1'] = [240, 213, 85, 60]
#imgs_bb_data[filenames[0]]['2'] = [440, 305, 66, 86]
imgs_bb_data[filenames[1]] = OD()
imgs_bb_data[filenames[1]]['1'] = [160, 148, 200, 460]

# Load image from disk
images = []

 # ++++ Saliency for whole image ++++ #
for i, file in enumerate(filenames):
    # Read all images and append them to a list
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    # plt.imshow(image)
    # plt.show()
    # display_saliency(file)

# ++++ Saliency for RoIs ++++ #
print(imgs_bb_data.items())
for i, (name, anns) in enumerate(imgs_bb_data.items()):
    print("\n" + "-" * 50)
    print("\nImage: {}".format(name))
    for j, bbox in anns.items():
        print("Annotation {}: {}".format(j, bbox))
        box_key = tuple(bbox)
        ground_truth[box_key] = extract_roi(images[i], bbox)
        plt.subplot(121)
        boxed_roi = draw_bbox(images[i], bbox, title="Original: RoI")
        save_image2disk(boxed_roi, output_dir+"boxed_roi"+name)
        plt.subplot(122)
        plt.imshow(ground_truth[box_key])
        plt.title('Tight BBox')
        plt.show()
        roi_file = output_dir + "roi_" + name
        save_image2disk(ground_truth[box_key], roi_file)
        display_saliency(roi_file, {'mbd':0.1, 'rbd':0.4})
    # ++++ Modify RoI ++++ #
    for j, bbox in anns.items():
        bbox = modify_roi(bbox, mod_factor)
     
    for j, bbox in anns.items():
        # Extract modified RoI
        print("To extract: {}".format(bbox))
        box_key = tuple(bbox)
        ground_truth[box_key] = extract_roi(images[i], bbox)
        plt.subplot(121)
        draw_bbox(images[i], bbox, title="Original: Expanded RoI", color=(255, 0, 0))
        plt.subplot(122)
        plt.imshow(ground_truth[box_key])
        plt.title('Expanded BBox: ' + str(mod_factor) + " %")
        plt.show()
        roi_mod_file = output_dir + "roi_mod_" + name
        save_image2disk(ground_truth[box_key], roi_mod_file)
        display_saliency(roi_mod_file, {'mbd':0.1, 'rbd':0.4})