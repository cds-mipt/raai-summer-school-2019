import os
import cv2
import numpy as np
from skimage.measure import label, regionprops
import time
from datetime import datetime
from Cvat import CvatDataset
from skimage.draw import line


class ScenePainter:
    def __init__(self):
        self._background_image = None
        self._background_path = 'data/background/target/DJI_AVC-0-02-22-916.jpg'
        self._cars_library_path = 'data/cars_selected'
        self._cars_library_list = []
        self._cars_image_path = 'data/cars/images'
        self._cars_mask_path = 'data/cars/masks'
        self._cars_out_path = 'data/cars/tmp'
        os.makedirs(self._cars_mask_path, exist_ok=True)
        self.cars_sizes = []
        self.bresenham_path_list = []

        self.path_to_cvat_paths = 'data/background/9_Crossroad_Sim_Routes.xml'

    def load_background(self, background_path=None):
        if (background_path is None):
            background_path = self._background_path
        else:
            self._background_path = background_path
        self._background_image = cv2.imread(background_path)
        return self._background_image

    def load_cars_library(self, cars_library_path=None):
        if(cars_library_path is None):
            cars_library_path = self._cars_library_path
        else:
            self._cars_library_path = cars_library_path
        #background_image = cv2.imread(background_path)
        for filename in sorted(os.listdir(cars_library_path)):
            is_valid = False
            white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
            for extension in white_list_formats:
                if filename.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                self._cars_library_list.append(filename)
        return self._cars_library_list

    def load_cars_sizes(self):
        self.cars_sizes = []
        for filename in self._cars_library_list:
            mask_fname = ('.').join(filename.split('.')[:-1])
            car_mask_image = cv2.imread(self._cars_mask_path + '/' + mask_fname + '.bmp')
            label_image = label(car_mask_image[:,:,0])
            n_all_regions = len(regionprops(label_image))
            region = regionprops(label_image)[0]
            minr, minc, maxr, maxc = region.bbox

            region_width = (maxc - minc)
            region_height = (maxr - minr)
            self.cars_sizes.append((region_width, region_height))

        return self.cars_sizes

    def load_car_paths(self):

        cvat_data = CvatDataset()
        cvat_data.load(self.path_to_cvat_paths)

        image_ids = cvat_data.get_image_ids()

        self.paths = []
        for idx, image in enumerate(image_ids):
            self.paths = cvat_data.get_polylines(image)

        return self.paths

    def load_expanded_car_paths(self, shift=100):

        paths = self.load_car_paths()
        for idx, path in enumerate(paths):
            image_w = self._background_image.shape[1]
            image_h = self._background_image.shape[0]

            start_point = path['points'][0]
            if start_point[0] == 0:
                start_point = [start_point[0]-shift, start_point[1]]
            if start_point[1] == 0:
                start_point = [start_point[0], start_point[1]-shift]
            if start_point[0] == image_w:
                start_point = [start_point[0]+shift, start_point[1]]
            if start_point[1] == image_h:
                start_point = [start_point[0], start_point[1]+shift]
            paths[idx]['points'] = [start_point]+paths[idx]['points']

            end_point = path['points'][-1]
            if end_point[0] == 0:
                end_point = [end_point[0] - shift, end_point[1]]
            if end_point[1] == 0:
                end_point = [end_point[0], end_point[1] - shift]
            if end_point[0] == image_w:
                end_point = [end_point[0] + shift, end_point[1]]
            if end_point[1] == image_h:
                end_point = [end_point[0], end_point[1] + shift]

            paths[idx]['points'] = paths[idx]['points'] + [end_point]
            self.paths = paths

        return self.paths

    def calc_angle(self, point_prev, point_next):
        vector_car = [point_next[0]-point_prev[0], point_next[1]-point_prev[1]]
        vector_base = [1.0, 0.0]
        scalar_mult = np.dot(vector_car, vector_base)

        vector_car_len = np.sqrt(vector_car[0] ** 2 + vector_car[1] ** 2)
        vector_base_len = 1
        cos = scalar_mult / (vector_car_len * vector_base_len)

        angle_rad = np.arccos(cos)
        if point_next[1]-point_prev[1] > 0:
            angle_rad = -angle_rad

        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def get_bresenham_paths(self, paths):
        self.bresenham_path_list = []
        for path in paths:
            points = path['points']
            bresenham_points = [None,None,None]
            for idx,point in enumerate(points):
                if idx == 0:
                    prev_point = point
                else:
                    current_point = point
                    xx, yy = line(int(prev_point[0]), int(prev_point[1]),
                                  int(current_point[0]), int(current_point[1]))
                    if bresenham_points[0] is None:
                        bresenham_points[0] = xx
                    else:
                        bresenham_points[0] = np.hstack((bresenham_points[0],xx))
                    if bresenham_points[1] is None:
                        bresenham_points[1] = yy
                    else:
                        bresenham_points[1] = np.hstack((bresenham_points[1],yy))

                    # angle calculation
                    angle_deg = self.calc_angle(prev_point, current_point)
                    if (idx < len(points)-1):
                        next_point = points[idx+1]
                        next_angle_deg = self.calc_angle(current_point, next_point)
                        if np.abs(next_angle_deg-angle_deg)<180:
                            aa = np.arange(len(xx)) * (next_angle_deg-angle_deg) / len(xx) + angle_deg
                        else:
                            if (next_angle_deg - angle_deg) < 0:
                                aa = np.arange(len(xx)) * (next_angle_deg - angle_deg + 360) / len(xx) + angle_deg
                            else:
                                aa = np.arange(len(xx)) * (next_angle_deg - angle_deg - 360) / len(xx) + angle_deg
                    else:
                        aa = np.ones((len(xx))) * angle_deg
                    if bresenham_points[2] is None:
                        bresenham_points[2] = aa
                    else:
                        bresenham_points[2] = np.hstack((bresenham_points[2], aa))

                    prev_point = current_point


            self.bresenham_path_list.append(bresenham_points)

        return self.bresenham_path_list



    def calc_rotation_matrix(self, in_image, in_angle, scale=1.0):

        center_rot = (int(in_image.shape[1] / 2), int(in_image.shape[0] / 2))

        rotation_mat = cv2.getRotationMatrix2D(center_rot, in_angle, scale)

        height = in_image.shape[0]
        width = in_image.shape[1]

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - center_rot[0]
        rotation_mat[1, 2] += bound_h / 2 - center_rot[1]

        return rotation_mat, (bound_w, bound_h)

    def show_car (self, x, y, angle, car_index=None,
                  background_image=None, full_mask_image=None, is_gaussian_shadow=True, verbose=False):

        if (background_image is None):
            background_image = self._background_image

        if (full_mask_image is None):
            full_mask_image = np.zeros((background_image.shape[0], background_image.shape[1]),dtype='uint8')

        if (car_index is None):
            car_index = np.random.randint(len(self._cars_library_list))

        car_image = cv2.imread(self._cars_image_path + '/'+self._cars_library_list[car_index])

        mask_fname = ('.').join(self._cars_library_list[car_index].split('.')[:-1])
        car_mask_image = cv2.imread(self._cars_mask_path + '/' + mask_fname + '.bmp')

        masked_image = cv2.bitwise_and(car_image, car_mask_image)
        if (verbose):
            cv2.imwrite(self._cars_out_path + '/show-' + str(car_index) + '-masked.jpg', masked_image)

        rotation_mat, (bound_w, bound_h) = self.calc_rotation_matrix(masked_image, angle)

        # rotate image with the new bounds and translated rotation matrix
        masked_image = cv2.warpAffine(masked_image, rotation_mat, (bound_w, bound_h))
        car_mask_image = cv2.warpAffine(car_mask_image, rotation_mat, (bound_w, bound_h))

        start_x = min(max(int(x-bound_w/2),0),background_image.shape[1])
        start_y = min(max(int(y-bound_h/2),0),background_image.shape[0])
        end_x = max(min(int(x + bound_w / 2), background_image.shape[1]),0)
        end_y = max(min(int(y + bound_h / 2), background_image.shape[0]),0)

        if start_x == end_x or start_y == end_y:
            if(verbose):
                cv2.imwrite(self._cars_out_path + '/show-' + str(car_index) + '.jpg', background_image)
            return background_image, full_mask_image

        background_crop = background_image[start_y:end_y,start_x:end_x]

        mask_start_x = start_x - int(x-bound_w/2)
        mask_start_y = start_y - int(y - bound_h / 2)
        mask_end_x = mask_start_x + end_x - start_x
        mask_end_y = mask_start_y + end_y - start_y


        mask_bool = (car_mask_image[mask_start_y:mask_end_y,mask_start_x:mask_end_x,0]).astype('uint8') #/255
        mask_inv = cv2.bitwise_not(mask_bool)#-254

        if is_gaussian_shadow:
            mask_shadow = mask_inv
            for i in range(2):
                mask_shadow = cv2.GaussianBlur(mask_shadow, (7, 7), 2)
            mask_shadow = (mask_shadow/255)
            mask_shadow = mask_shadow.reshape((mask_shadow.shape[0],mask_shadow.shape[1],1))
            background_crop = (background_crop * mask_shadow).astype('uint8')

        masked_image = masked_image[mask_start_y:mask_end_y,mask_start_x:mask_end_x]
        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(background_crop, background_crop, mask=mask_inv)
        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(masked_image, masked_image, mask=mask_bool)
        if(verbose):
            cv2.imwrite(self._cars_out_path + '/show-' + str(car_index) + '-affine.jpg', img1_bg)
            cv2.imwrite(self._cars_out_path + '/show-' + str(car_index) + '-affine-mask.jpg', img2_fg)
        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        background_image[start_y:end_y,start_x:end_x] = dst
        full_mask_image[start_y:end_y,start_x:end_x] = full_mask_image[start_y:end_y,start_x:end_x] + mask_bool
        if(verbose):
            cv2.imwrite(self._cars_out_path+'/show-'+str(car_index)+'.jpg', background_image)

        return background_image, full_mask_image

    def get_cars_library_path(self):
        return self._cars_library_path

    def get_background_path(self):
        return self._background_path

    def get_cars_library_list(self):
        return self._cars_library_list

    def set_cars_image_path(self, cars_image_path):
        self._cars_image_path = cars_image_path

    def get_cars_image_path(self):
        return self._cars_image_path

    def set_cars_mask_path(self, cars_mask_path):
        self._cars_mask_path = cars_mask_path

    def get_cars_mask_path(self):
        return self._cars_mask_path

    def set_cars_out_path(self, cars_out_path):
        self._cars_out_path = cars_out_path

    def get_cars_out_path(self):
        return self._cars_out_path

# painter = ScenePainter()
# background_image = painter.load_background()
# painter.load_cars_library()
# car_sizes = painter.load_cars_sizes()
#
# background_image, mask_image = painter.show_car(x=background_image.shape[1]/2+20, y=background_image.shape[0]/2-35, angle=135,
#                  car_index=None, background_image=None)
# background_image, mask_image = painter.show_car(x=background_image.shape[1]/2-20, y=background_image.shape[0]/2+30, angle=-60,
#                  car_index=None, background_image=None)
# background_image, mask_image = painter.show_car(x=10, y=background_image.shape[0]/2+30, angle=0,
#                  car_index=None, background_image=None)
# background_image, mask_image = painter.show_car(x=background_image.shape[1]-200, y=background_image.shape[0]/2-30, angle=180,
#                  car_index=None, background_image=None)
# background_image, mask_image = painter.show_car(x=background_image.shape[1]-400, y=background_image.shape[0]/2-30, angle=180,
#                  car_index=None, background_image=None)
# background_image, mask_image = painter.show_car(x=background_image.shape[1]/2+40, y=background_image.shape[0]/2-200, angle=90,
#                  car_index=None, background_image=None)
# background_image, mask_image = painter.show_car(x=background_image.shape[1]/2-40, y=background_image.shape[0]/2+200, angle=-90,
#                  car_index=None, background_image=None)
# background_image, mask_image = painter.show_car(x=background_image.shape[1]/2+42, y=background_image.shape[0]/2-600, angle=90,
#                  car_index=None, background_image=None)
# background_image, mask_image = painter.show_car(x=background_image.shape[1]/2-45, y=background_image.shape[0]/2+700, angle=-90,
#                  car_index=None, background_image=None)
# background_image, mask_image = painter.show_car(x=background_image.shape[1]/2+300, y=background_image.shape[0]/2+35, angle=0,
#                  car_index=None, background_image=None)
#
# t=datetime.fromtimestamp(time.time())
# time_str = t.strftime("%Y-%m-%d-%H-%M-%S")
#
# #time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.time())
# out_path = painter.get_cars_out_path()+'/show-' + time_str + '.jpg'
# cv2.imwrite(out_path, background_image)
