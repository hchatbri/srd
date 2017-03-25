import numpy as np
from ImageUtilities import extractContours, euclidean_distance
import math


class SRD():

    FEATURE_COMPONENT_THRESHOLD = 0.125

    ''' Constructor:
        - Finds foreground pixels
        - Finds the object's center of mass
    '''
    def __init__(self, img):
        self.foreground_pixels = []
        self.feature_histogram = np.zeros((256, 256), dtype=np.float64)
        self.centroid_pixel = [0, 0]

        img = extractContours(img)
        height, width = img.size
        pixels = img.load()

        # find foreground pixels
        for i in range(height):
            for j in range(width):
                value = pixels[i, j]
                if value < 255:
                # if SRD.isForeground(img, i, j):
                    self.foreground_pixels.append((i, j))
        # find the object center of mass
        for p in self.foreground_pixels:
            self.centroid_pixel[0] = self.centroid_pixel[0] + p[0]
            self.centroid_pixel[1] = self.centroid_pixel[1] + p[1]
            self.centroid_pixel[0] /= len(self.foreground_pixels)
            self.centroid_pixel[1] /= len(self.foreground_pixels)

    ''' Extract features '''
    def extract_features(self):
        for p in self.foreground_pixels:
            # in the original SRD, the average distance from p to all other foreground pixels is calculated
            # here, the distance here is calculated from p to the centroid, for the sake of efficiency
            p_distance_to_centroid = euclidean_distance(p, self.centroid_pixel)
            near_region = np.zeros(8)
            far_region = np.zeros(8)
            for q in self.foreground_pixels:
                p_q_angle = math.atan2(q[1] - p[1], q[0] - p[0])
                p_q_angle = round(math.degrees(p_q_angle), 2)
                if p_q_angle < 0:
                    p_q_angle += 360

                angle_index = int(p_q_angle / 45)

                if euclidean_distance(p, q) < p_distance_to_centroid:
                    near_region[angle_index] += 1
                else:
                    far_region[angle_index] += 1

            # now we have the 2 feature components for p.
            # normalize them
            if sum(near_region) > 0:
                near_region /= sum(near_region)
            if sum(far_region) > 0:
                far_region /= sum(far_region)
            # threshold and convert them to decimal
            near_region_decimal = 0
            far_region_decimal = 0
            for i in range(len(near_region)):
                if near_region[i] < SRD.FEATURE_COMPONENT_THRESHOLD:
                    near_region[i] = 0
                else:
                    near_region[i] = 1

                if far_region[i] < SRD.FEATURE_COMPONENT_THRESHOLD:
                    far_region[i] = 0
                else:
                    far_region[i] = 1

                near_region_decimal += near_region[i] * math.pow(2, i)
                far_region_decimal += far_region[i] * math.pow(2, i)
            # then use them to fill in the 256x256 histogram
            self.feature_histogram[int(near_region_decimal)][int(far_region_decimal)] += 1

        self.feature_histogram /= len(self.foreground_pixels)


    ''' Returns the distance between two images, using the histogram intersection distance '''
    ''' The distance is in [0, 1]. The smaller it gets, the more similar the images are '''
    @staticmethod
    def distance(descriptor_1, descriptor_2):
        d = 0
        for i in range(descriptor_1.feature_histogram.shape[0]):
            for j in range(descriptor_1.feature_histogram.shape[1]):
                d += min(descriptor_1.feature_histogram[i][j], descriptor_2.feature_histogram[i][j])

        return 1 - d
