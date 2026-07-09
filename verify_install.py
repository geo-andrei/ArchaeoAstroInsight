import sys
print("Python:", sys.version)

import numpy as np; print("NumPy:", np.__version__)
import cv2; print("OpenCV:", cv2.__version__)
import pandas as pd; print("Pandas:", pd.__version__)
import sklearn; print("Scikit-learn:", sklearn.__version__)
import matplotlib; print("Matplotlib:", matplotlib.__version__)
import scipy; print("SciPy:", scipy.__version__)

import torch, torchvision
print("Torch:", torch.__version__, "| Torchvision:", torchvision.__version__)

import tensorflow as tf
from tensorflow import keras
print("TensorFlow:", tf.__version__, "| Keras:", keras.__version__)

from ultralytics import YOLO
print("Ultralytics:", YOLO.__name__)
print("OK: all imports succeeded.")
