import matplotlib.pyplot as plt
import numpy as np

setting = '0'
pred1 = np.load('./res/'+setting+'/finalreal_prediction.npy')
true = np.load('./res/'+setting+'/real_prediction.npy')
print(pred1.shape)
print(true.shape)



plt.figure()
plt.plot(true[0,:,-1], color='red', label='Informer')
plt.plot(pred1[0,:,-1],color='blue',  label='PCRNet')
plt.legend()
plt.show()
