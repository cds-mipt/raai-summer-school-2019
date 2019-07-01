from cvat import CvatDataset

import os
import cv2
import numpy as np

class SourcePreparation:

    def __init__(self):

        self.path_to_cvat_data = 'data/background/7_RAAI_Summer_School_2.xml'

        self.background_clear_path = 'data/background/clear/'
        self.background_target_path = 'data/background/target/'
        self.background_source_path = 'data/background/source/'

        self.cars_images_path = 'data/cars/images/'
        self.cars_masks_path = 'data/cars/masks/'


    def cross_lines(self, x1_1, y1_1, x1_2, y1_2,
                    x2_1, y2_1, x2_2, y2_2):

        def check_point(x, y):
            if min(x1_1, x1_2) <= x <= max(x1_1, x1_2):
                print('Точка пересечения отрезков есть, координаты: ({0:f}, {1:f}).'.
                      format(x, y))
                return (x, y)
            else:
                print('Точки пересечения отрезков нет.')

        A1 = y1_1 - y1_2
        B1 = x1_2 - x1_1
        C1 = x1_1 * y1_2 - x1_2 * y1_1
        A2 = y2_1 - y2_2
        B2 = x2_2 - x2_1
        C2 = x2_1 * y2_2 - x2_2 * y2_1

        if B1 * A2 - B2 * A1 and A1:
            y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
            x = (-C1 - B1 * y) / A1
            result = check_point(x, y)
        elif B1 * A2 - B2 * A1 and A2:
            y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
            x = (-C2 - B2 * y) / A2
            result = check_point(x, y)
        else:
            print('Точки пересечения отрезков нет, отрезки ||.')
            result = None

        return result

    def background_alignment(self, points_list, image_size, image_name):
        axis_vertical_p_cnt = 0
        axis_horizontal_p_cnt = 0
        for point in points_list:
            if point["label"] == "axis_vertical_p":
                if (axis_vertical_p_cnt == 0):
                    x1_1 = point["points"][0][0]
                    y1_1 = point["points"][0][1]
                    axis_vertical_p_cnt = 1
                else:
                    x1_2 = point["points"][0][0]
                    y1_2 = point["points"][0][1]
            if point["label"] == "axis_horizontal_p":
                if (axis_horizontal_p_cnt == 0):
                    x2_1 = point["points"][0][0]
                    y2_1 = point["points"][0][1]
                    axis_horizontal_p_cnt = 1
                else:
                    x2_2 = point["points"][0][0]
                    y2_2 = point["points"][0][1]
        result = self.cross_lines(x1_1, y1_1, x1_2, y1_2,
                             x2_1, y2_1, x2_2, y2_2)
        if result is not None:
            width = image_size["width"]
            height = image_size["height"]
            center = (result[0], result[1], 1)
            center_rot = (int(width / 2), int(height / 2))
            angle = np.arctan((y2_1 - y2_2) / (x2_1 - x2_2)) / np.pi * 180

            full_image_path = self.background_clear_path + os.path.basename(image_name)

            img_bgr = cv2.imread(full_image_path)

            rotation_mat = cv2.getRotationMatrix2D(center_rot, angle, 1.0)

            # rotation calculates the cos and sin, taking absolutes of those.
            abs_cos = (rotation_mat[0, 0])
            abs_sin = (rotation_mat[0, 1])

            # find the new width and height bounds
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            # subtract old image center (bringing image back to origo) and adding the new image center coordinates
            rotation_mat[0, 2] += bound_w / 2 - center_rot[0]
            rotation_mat[1, 2] += bound_h / 2 - center_rot[1]

            # rotate image with the new bounds and translated rotation matrix

            img_bgr = cv2.warpAffine(img_bgr, rotation_mat, (bound_w, bound_h))
            center_points = np.transpose(np.asarray([center]))
            new_center = np.dot(rotation_mat, center_points)

            # calc background image with square size
            distance_left = int(new_center[0])
            distance_right = bound_w - distance_left

            distance_top = int(new_center[1])
            distance_bottom = bound_h - distance_top

            size_r = int(min(distance_left, distance_right, distance_top, distance_bottom))
            background_bgr = img_bgr[(distance_top - size_r):(distance_top + size_r),
                             (distance_left - size_r):(distance_left + size_r)]

            # background image saving
            full_target_path = self.background_target_path + os.path.basename(image_name)

            cv2.imwrite(full_target_path, background_bgr)

    def cars_cropping(self, car_polygons, image_name):
        full_source_path = self.background_source_path + os.path.basename(image_name)
        img_src_bgr = cv2.imread(full_source_path)

        height = img_src_bgr.shape[0]
        width = img_src_bgr.shape[1]

        for car_idx, polygon in enumerate(car_polygons):

            color = (0, 0, 255)
            is_right_car = False
            if (polygon["label"] == "car_right"):
                color = (0, 255, 0)
                is_right_car = True

            polygon_points = np.asarray(polygon["points"], dtype='int32')
            rect = cv2.minAreaRect(polygon_points)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(img_src_bgr, [box], 0, color, 2)
            ellipse = cv2.fitEllipse(polygon_points)
            # cv2.ellipse(img_src_bgr, ellipse, (255, 255, 255), 1)

            # cropping cars
            car_center_x = ellipse[0][0]
            car_center_y = ellipse[0][1]
            car_e_a = ellipse[1][0]
            car_e_b = ellipse[1][1]
            car_r_max = max(car_e_a, car_e_b) / 2 * 1.15
            car_r_min = min(car_e_a, car_e_b) / 2 * 1.4
            car_angle = ellipse[2]
            start_y = int(max(car_center_y - car_r_max, 0))
            end_y = int(min(car_center_y + car_r_max, height))
            start_x = int(max(car_center_x - car_r_max, 0))
            end_x = int(min(car_center_x + car_r_max, width))

            cropped_car = img_src_bgr[start_y:end_y, start_x:end_x]

            center_cropped_car = (int(cropped_car.shape[0] / 2), int(cropped_car.shape[1] / 2))
            if (is_right_car):
                angle = car_angle - 90
            else:
                angle = car_angle + 90

            rotation_mat = cv2.getRotationMatrix2D(center_cropped_car, angle, 1.0)

            bound_w = int(car_r_max) * 2
            bound_h = int(car_r_min) * 2

            # subtract old image center (bringing image back to origo) and adding the new image center coordinates
            rotation_mat[0, 2] += bound_w / 2 - center_cropped_car[0]
            rotation_mat[1, 2] += bound_h / 2 - center_cropped_car[1]

            # rotate image with the new bounds and translated rotation matrix
            cropped_car = cv2.warpAffine(cropped_car, rotation_mat, (bound_w, bound_h))

            full_car_target_path = self.cars_images_path + os.path.basename(image_name) + '_' + str(car_idx) + '.jpg'

            cv2.imwrite(full_car_target_path, cropped_car)

            # polygon mask creation
            shift = np.asarray([[start_x, start_y]])
            polygon_points = polygon_points - shift
            polygon_points = np.hstack((polygon_points, np.ones((polygon_points.shape[0], 1), dtype=polygon_points.dtype)))
            polygon_points_transformed = np.dot(rotation_mat, np.transpose(polygon_points))

            mask = np.zeros(cropped_car.shape, dtype=np.uint8)
            # roi_corners1 = np.array([[(10, 10), (300, 300), (10, 300)]], dtype=np.int32)
            roi_corners = np.asarray([np.transpose(polygon_points_transformed)], dtype=np.int32)
            # fill the ROI so it doesn't get wiped out when the mask is applied
            channel_count = cropped_car.shape[2]  # i.e. 3 or 4 depending on your image
            car_mask_color = (255,) * channel_count
            cv2.fillPoly(mask, roi_corners, car_mask_color)
            # from Masterfool: use cv2.fillConvexPoly if you know it's convex

            # apply the mask
            # masked_image = cv2.bitwise_and(image, mask)
            full_car_mask_path = self.cars_masks_path + os.path.basename(image_name) + '_' + str(car_idx) + '.bmp'

            cv2.imwrite(full_car_mask_path, mask)

    def all_resources_preparation(self):
        path_to_cvat_data = self.path_to_cvat_data
        #target_path = self.background_target_path

        cvat_data = CvatDataset()
        cvat_data.load(path_to_cvat_data)

        image_ids = cvat_data.get_image_ids()

        polygons = []
        points = []
        image_sizes = []
        image_names = []
        for idx, image in enumerate(image_ids):
            polygons.append(cvat_data.get_polygons(image))
            points.append(cvat_data.get_points(image))
            image_sizes.append(cvat_data.get_size(image))
            image_names.append(cvat_data.get_name(image))

            points_list = points[idx]
            image_size = image_sizes[idx]
            image_name = image_names[idx]

            # background creation
            self.background_alignment(points_list=points_list, image_size=image_size, image_name=image_name)

            # cars cropping
            car_polygons = polygons[idx]
            self.cars_cropping(car_polygons=car_polygons, image_name=image_name)

        print(image_ids)

source_preparator = SourcePreparation()
source_preparator.all_resources_preparation()






