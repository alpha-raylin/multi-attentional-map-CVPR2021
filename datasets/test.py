from ctypes import sizeof
import os
import json
import dataset

datalist = dataset.DeepfakeDataset(datalabel='ff-all-c23-jpeg')
image, label = datalist[20]
print(image.shape)