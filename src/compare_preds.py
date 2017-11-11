import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import scipy.io

# path = "~/Documents/Spacebergs/src/input/"
# gbm_scores = pd.read_csv(path + "gbm_preds.csv")
# cnn_scores = pd.read_csv(path + "cnn_scores.csv")



# gbm_scores = pd.read_csvc

path = "/home/emilal/Documents/Spacebergs/src/input"
num_pixels = scipy.io.loadmat(path + '/band1_numPixels.mat')
