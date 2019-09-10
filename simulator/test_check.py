import gym
import gym_car_intersect
import matplotlib
import matplotlib.pyplot as plt
from sac_torch.env_wrappers import SubprocVecEnv, make_pixel_env

env = gym.make("CarIntersect-v3")
currentImage = env.reset()


#===INSERT IMAGE TO COORD===========================================
from skimage.measure import label, regionprops

# loading recognition models
self.recognized_cars_count = 0
self.segmentation_model_path = 'unetdetector/models/unet2019-07-02-22-08-09.29-tloss-0.0171-tdice-0.9829.hdf5'
self.segmentation_image_width = 512
self.segmentation_image_height = 512
self.segmentation_model = unet_light(pretrained_weights=self.segmentation_model_path,
                                     input_size=(self.segmentation_image_width,
                                     self.segmentation_image_height, 3))

# Recognition process
self.recognized_state = []
start_time = time.time()
input_image = cv2.cvtColor(self.currentImage, cv2.COLOR_BGR2RGB)
input_image = input_image / 255.0

net_input_image = cv2.resize(input_image,
                             (self.segmentation_image_width, self.segmentation_image_height))
net_input_image = np.reshape(net_input_image, (1,) + net_input_image.shape)

segmentation_result = self.segmentation_model.predict(net_input_image)
segmentation_mask = segmentation_result[0][:, :, 0]
binary_mask = ((segmentation_mask > 0.9) * 255).astype('uint8')

self.maskImage = cv2.resize(binary_mask,
                            (self.currentImage.shape[1], self.currentImage.shape[0]))

label_image = label(binary_mask)

# cropped_cars = []
scale = self.currentImage.shape[0] / net_input_image.shape[1]
area_threshold = 100
for region in regionprops(label_image):
    start_y, start_x, end_y, end_x = region.bbox
    start_y = int(start_y * scale)
    start_x = int(start_x * scale)
    end_y = int(end_y * scale)
    end_x = int(end_x * scale)
    if ((end_y - start_y) * (end_x - start_x) > area_threshold):
        car_points = region.coords
        ellipse = cv2.fitEllipse(car_points)
        car_angle_mask = ellipse[2]
        car_box = [start_x, start_y, end_x, end_y, car_angle_mask]
        # bboxes.append(car_box)
        cv2.rectangle(self.currentImage,
                      (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        # cropped_cars.append(input_image[start_y:end_y, start_x:end_x])

        # angle calculation based on previous recognition
        car_id, car_prev_center, car_prev_angle = self.findInPreviousState(car_box)
        if car_id is not None:
            car_new_center = (int((start_x + end_x) / 2), int((start_y + end_y) / 2))
            car_angle = self.painter.calc_angle(car_prev_center, car_new_center)
            if car_angle is None:
                car_angle = car_prev_angle
            car_box[4] = car_angle
            car_box.append(car_id)
        else:
            self.recognized_cars_count += 1
            car_box.append(self.recognized_cars_count)
        text = str(car_box[-1]) + ' : ' + str(int(car_box[-2]))  # id : angle#
        cv2.putText(self.currentImage, text, (start_x, start_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)
        self.recognized_state.append(car_box)
self.prev_recognized_state = self.recognized_state

agent_car_id = 3 #id of agent car is always 3
scale_factor = 80/self.backgroundImage.shape[1]

curr_state_coord = []
for car in self.recognized_state:
    car_x_center = ((car[2] + car[0])/2 - self.backgroundImage.shape[1]/2)*scale_factor
    car_y_center = ((car[3] + car[1]) / 2 - self.backgroundImage.shape[0]/2)*scale_factor*(-1)
    car_angle = np.radians(car[4] - 90)
    car_idx = car[5]
    if car_idx == agent_car_id:
        curr_state_coord = np.hstack(([car_x_center, car_y_center, car_angle], curr_state_coord))
    else:
        curr_state_coord = np.hstack((curr_state_coord, [car_x_center, car_y_center, car_angle]))
if len(curr_state_coord) == 12:
    self.curr_state_coord = curr_state_coord




# env = SubprocVecEnv([lambda: make_pixel_env("CarIntersect-v3") for _ in range(3)])
# # env = make_pixel_env("CarIntersect-v3")
# s = env.reset()
# plt.imshow(s[3])
# env.action_space
# env.observation_space
# # import torch
# # import torch.nn as nn
# #
# # net = nn.Sequential(nn.Linear(3*1378**2, 3*4), nn.Sigmoid())
# #
# # from utils import *
# # env = make_env("CarIntersect-v2", net, "torch")
# # print(env.reset())
#
# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# done = False
# obj = ax.imshow(s[1])
#
# for i in range(100):
#     if done:
#         env.reset()
#     s, r, done, _ = env.step([0, 1, 0])
#     obj.set_data(s[1])
#     fig.canvas.draw()
#     plt.imshow(s[1])
#     plt.show()

# done = False
# for i in range(100):
#     if done: break
#     s, r, done, _ = env.step([0, 1, 0])
#     plt.imshow(s[0])
#     plt.show()
#     # for img in s:
#     #     plt.imshow(img)
#     #     plt.show()
