import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

# import sys
# np.set_printoptions(threshold=sys.maxsize)

def show(*imgs):
    fig=plt.figure(figsize=(20, 20))
    columns = math.ceil(np.sqrt(len(imgs)))
    rows = math.ceil(len(imgs)/columns)
    i = 1
    for img in imgs:
        fig.add_subplot(rows,columns,i)
        plt.imshow(img)
        i+=1
    plt.show()

def get_extract(img, output, i):
    mask_i = np.zeros((output.shape))
    mask_i[ output == i ] = 1
    mask_i = mask_i.astype(np.uint8)

    kernel = np.ones((3,3), np.uint8) 
    mask_i = cv2.dilate(mask_i, kernel, iterations=1) 

    extract = img * mask_i
    return extract

def crop(stat, extract):
    left, top, width, height, area = stat
    extract = np.pad(extract, pad_width=14, mode='constant', constant_values=0)
    x = int(left + (width/2))
    y = int(top + (height/2))
    cropped = extract[y: y+28, x: x+28]
    return cropped

def segment(img, stride = 20):
    digits = []
    areas = []
    widths = []
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity = 4)
    for i in range(1,nb_components):
        left, top, width, height, area = stats[i]
        extracted_digit = get_extract(img, output, i)

        # # Debug Statement
        # print(stats[i])
        # show(extracted_digit)
        if area < 45:
            continue

        join = 0

        # If 2 digits overlap
        if width >= 22:
            nb_components_j = 2
            level = 150
            bad_components = 0
            while level < 255 and nb_components_j-bad_components < 3:
                bad_components = 0
                _, thresh = cv2.threshold(extracted_digit, level, 255, cv2.THRESH_TOZERO)
                nb_components_j, output_j, stats_j, centroids_j = cv2.connectedComponentsWithStats(thresh, connectivity = 4)

                # print(level, nb_components_j)
                # extracts = []

                for j in range(1,nb_components_j):
                    left, top, width, height, area = stats_j[j]

                    # extracted_digit = get_extract(img, output_j, j)
                    # extract = crop(stats_j[j], extracted_digit)
                    # extracts.append(extract)
                    # print(area)

                    if area < 45:
                        bad_components += 1
                # show(*extracts)
                # print(bad_components)
                level += stride

            if nb_components_j >= 3:
                join = 1
                for j in range(1, nb_components_j):
                    left, top, width, height, area = stats_j[j]
                    if area < 45:
                        continue
                    extracted_digit = get_extract(img, output_j, j)
                    digit = crop(stats_j[j], extracted_digit)
                    widths.append(width)
                    areas.append(area)
                    digits.append(digit)
        
        if not join:
            # Crop
            digit = crop(stats[i], extracted_digit)
            widths.append(width)
            areas.append(area)
            digits.append(digit)
    return digits, areas, widths

if __name__ == "__main__":
    imgs = np.load('Data/data0.npy')

    img = imgs[6]
    show(img)
    digits, _, _= segment(img)
    show(*digits)

    # anom = []
    # for i in range(0,10000):
    #     img = imgs[i]
    #     digits = segment(img)
    #     if len(digits) != 4:
    #         anom.append(i)
    # print(anom)
    # print(len(anom))
    # anom_imgs = [imgs[i] for i in anom]
    # show(*anom_imgs[:10])
