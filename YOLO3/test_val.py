import numpy as np
stride = [8, 16, 32]
strides = np.array(stride)
bbox_xywh = [1, 2, 3, 4]
bbox_xywh = np.array(bbox_xywh)

a = strides[:, np.newaxis]
print(a)

#print(bbox_xywh)
b = bbox_xywh[np.newaxis, :]
print(b)

c = b / a
print(c)