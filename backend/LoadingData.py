# based off https://www.kaggle.com/code/architkhatri/before-and-after-over-fitting-comparison-cnn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from PIL import Image
import os
print(os.listdir("./input"))
print(os.listdir("./input/car_data"))
