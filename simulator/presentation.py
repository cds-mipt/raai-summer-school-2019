import sys
import os
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QMainWindow
from PyQt5.QtWidgets import QLabel, QLineEdit
from PyQt5.QtGui  import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer
import cv2
from ScenePainter import ScenePainter
from shapely.geometry import Polygon
from CrossroadSimulatorGUI import CrossroadSimulatorGUI

import time

import gym
import gym_car_intersect

# # Simple agent, which make random actions:
# class RandomAgent:
#     def __init__(self):
#         self.env = gym.make("CarIntersect-v1")
#
#     def __call__(self, state):
#         return self.env.action_space.sample()
#
# agent = RandomAgent()


from stable_baselines import DQN

class DQNAgent:
    def __init__(self, path="stable_baseline/saved_models/dqn/best_model.pkl"):
        self.model = DQN.load(path)

    def __call__(self, state):
        action, _ = self.model.predict(state)
        return action

agent = DQNAgent()


app = QApplication(sys.argv)
ex = CrossroadSimulatorGUI(agent)
ex.move(0, 0)
ex.show()

sys.exit(app.exec_())
