import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('data.csv')
x = data['time']
y1 = data['angle_set0']
y2 = data['angle_set1']
y3 = data['angle_real0']
y4 = data['angle_real1']
plt.plot(x, y1, c='r', label='angle_set0')
plt.plot(x, y2, c='g', label='angle_set1')
plt.plot(x, y3, c='b', label='angle_real0')
plt.plot(x, y4, c='c', label='angle_real1')
plt.xlabel('Time/s')
plt.ylabel('Degree/°')

plt.legend()
plt.show()