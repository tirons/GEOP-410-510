import numpy as np
import matplotlib.pyplot as plt 
import sys 

WFM = np.loadtxt(sys.argv[1], skiprows=2, delimiter=',',usecols=[0,1])
plt.plot(WFM[:,0], 100*WFM[:,1])
plt.show()
