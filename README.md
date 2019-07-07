
## Demo solution of Hackathon 1. “Computer vision and reinforcement learning”

### Task description

- **[Task presentation](https://yadi.sk/i/S9dYSvqOCHTwWA)**

- **[Instructions on task implementation on MIPT Server](https://docs.google.com/document/d/12R8UmFpnrWTzCJxO73qNU_XZx5z4x88b9QhQPIC88vc/edit?usp=sharing)**

- **[Telegram Channel for Questions](https://t.me/joinchat/GdNiZhcY5fGQepbVrHJ6LQ)**

In the modern world, the creation of smart roads and smart cities through which unmanned vehicles drive is being actively researched.
Driving a car in autonomous mode through a crossroad is one of the objectives of such research. Obtaining information about the traffic situation at the crossroad is possible using sensors installed on the car, and information from external systems. Such an external system can be an unmanned aerial vehicle (UAV), hovering above the crossroad and having the possibility of reliable recognition of road users.

The task of planning the movement of an autonomous vehicle through such an crossroad can be solved either on the basis of pre-programmed scenarios or using a prospective approach based on deep reinforcement learning.

A special and high role in the development, debugging and research of artificial intelligence systems using computer vision and reinforcement learning methods is played by computer simulators.

The solution of the hackathon task involves working with an crossroad simulator created in the laboratory of cognitive dynamic systems of MIPT based on a UAV video and creating the best quality algorithm for driving an autonomous car through an crossroad without collisions with other vehicles.

![Scheme](https://github.com/cds-mipt/raai-summer-school-2019/blob/master/readme_files/Scheme-EN.png)

### Benefits from participation
- Learn how to prepare data for training of neural networks that detect objects and highlight them along the contour.
- Learn to apply the reinforcement learning approach to traffic planning.
- Learn how to work with popular deep learning libraries Keras and Tensorflow for solving computer vision tasks and pyTorch for reinforcement learning.
- Learn how to debug a Python program on a server with GPU on video cards with support for Nvidia CUDA technology.
- Develop a prototype of program to control the unmanned vehicle at the crossroad.
- The winners of the hackathon will have the opportunity to undergo an internship at [the Laboratory of Cognitive Dynamic Systems of MIPT](https://mipt.ru/science/labs/cognitive-dynamic-systems/), as well as give an advantage in entering [the new master's program “Methods and Technologies of Artificial Intelligence”](https://mipt.ru/education/departments/fpmi/master/methods-technologies-ai), which opens in 2019 at MIPT.

### Datasets

- **Training dataset for car segmentation and detection  on images:** [Link to training dataset .zip archive](https://yadi.sk/d/nb_kC-DmGcqoqA)

Dataset includes 500 color images, masks and bounding boxes and contains 3 folders:

- 'color' - folder with .jpg source color images
- 'mask' - folder with .bmp mask images with ground truth segmentation
- 'label' - folder with ground truth bounding boxes in .txt files 

In .txt files each line is in the format: ```<class_name> <left> <top> <right> <bottom>```.
  
E.g. The ground truth bounding boxes of the image "2008_000034.jpg" are represented in the file "2008_000034.txt": 
```  
car 6 234 45 362
car 1 156 103 336
car 36 111 198 416
car 91 42 338 500
```

- **Testing dataset for car segmentation and detection  on images:** [Link to test dataset with color images](https://yadi.sk/d/NxaCr9Yzmvr-GQ)
 
### Metrics
The organizers analyze 3 quality metrics of the participant decisions:

**1) Average Precision (AP) quality measure for cars detection implemented by participants compared to the reference labeling of a test sample**. To do this, organisers will use the open source utility [https://github.com/rafaelpadilla/Object-Detection-Metrics] (should be maximized);

The output of car detection algorithm should be folder of .txt files with labels for given images in the format:
```
<class_name> <confidence> <left> <top> <right> <bottom>
```
For example, file "2008_000034.txt": 
```
car 0.99001 80 1 295 500  
car 0.12601 36 13 404 316  
car 1.00000 430 117 500 307  
car 0.14585 212 78 292 118  
car 0.070565 388 89 500 196 
```
  
**2) reward to autonomous car for the reinforcement learning task**,

**3) the quality of the solution presentation**.

The teams demonstrate their decisions in the form of a presentation about the features of the technical implementation, indicating the prospects for the application (from 16:00 on July 7, 2019).

**The team with the highest total score wins.**

**The final total score is formed by the formula:**

I = A – (0.4ꞏN1+0.5ꞏN2+0.1ꞏN3), 

where A – total number of teams, Ni – occupied place in the rating on the i-th quality metric.

### Schedule

13:00 5 July, Phystech.Arctic, 4 floor, Room 4.24 - Team building from registered participants, server access grainting, task consultations

10:30 - 14:00, 6 July, Phystech.Arctic, 4 floor, Room 4.24 - Consultations

10:00 - 13:00, 7 July, Phystech.Arctic, 4 floor, Room 4.24 - Consultations

13:00 7 July, Phystech.Arctic, 4 floor, Room 4.24 - Running of solutions on test data

16:00 7 July, Phystech.Arctic Lecture Hall, 4 floor, Room 4.20 - Presentation of solutions

### Results

| Team | AP of car detection | RL reward | Presentation | Total score |
| :--- | :--- | :--- | :--- | :--- | 
| team01 | - | - | - | -  | -  | 
| team02 | - | - | - | -  | -  | 
| team03 | - | - | - | -  | -  | 
| team04 | - | - | - | -  | -  | 
| team05 | - | - | - | -  | -  | 
| team06 | - | - | - | -  | -  | 
| team07 | - | - | - | -  | -  | 
| team08 | - | - | - | -  | -  | 
| team09 | - | - | - | -  | -  | 
| team10 | - | - | - | -  | -  | 
