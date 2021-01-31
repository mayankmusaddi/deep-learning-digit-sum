import numpy as np
import matplotlib.pyplot as plt


train_data1 = np.load('data0.npy')
train_lab1 = np.load('lab0.npy')
# print(len(train_data1[0]))
for i in range(1,20):
	plt.imshow(train_data1[i])
	plt.savefig(str(i)+'img.png')
print(train_lab1[i])