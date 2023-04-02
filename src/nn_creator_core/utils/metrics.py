import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime


def distance(true, pred):
    x_true, y_true = true
    x_pred, y_pred = pred
    d = np.sqrt((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2)
    return d

