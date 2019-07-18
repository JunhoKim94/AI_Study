import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = pd.read_csv("./heart/Heart_Train.csv")
a = np.array(a)
print(a)