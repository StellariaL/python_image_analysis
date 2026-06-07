'''
Template for multiple plots
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

x=[]
y=[]

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs=axs.flatten()

axs[0].plot(x, y)

axs[0].set_title('')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()

axs[1].scatter(x, y)


plt.title('')
plt.tight_layout()
plt.show()
