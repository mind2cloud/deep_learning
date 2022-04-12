# importing libraries
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt

# defining the columns using normal distribution 

# column 1 
point1 = abs(np.random.normal(1, 12, 100))
# column 2 
point2 = abs(np.random.normal(2, 8, 100))
# column 3 
point3 = abs(np.random.normal(3, 2, 100))
# column 4 
point4 = abs(np.random.normal(10, 15, 100))

# x contains the features of our dataset 
# the points are concatenated horizontally 
# using numpy to form a feature vector. 
x = np.c_[point1, point2, point3, point4]

# the output labels vary from 0-3 
y = [int(np.random.randint(0, 4)) for i in range(100)]

# defining a pandas data frame to save 
# the data for later use 
data = pd.DataFrame()

# defining the columns of the dataset 
data['col1'] = point1
data['col2'] = point2
data['col3'] = point3
data['col4'] = point4

# plotting the various features (x) 
# against the labels (y). 
plt.subplot(2, 2, 1)
plt.title('col1')
plt.scatter(y, point1, color='r', label='col1')

plt.subplot(2, 2, 2)
plt.title('Col2')
plt.scatter(y, point2, color='g', label='col2')

plt.subplot(2, 2, 3)
plt.title('Col3')
plt.scatter(y, point3, color='b', label='col3')

plt.subplot(2, 2, 4)
plt.title('Col4')
plt.scatter(y, point4, color='y', label='col4')

# saving the graph 
plt.savefig('data_visualization.jpg')

# displaying the graph 
plt.show() 