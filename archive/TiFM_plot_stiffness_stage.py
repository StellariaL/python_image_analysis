import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu

filename='D:/Xiong Lab/force-displacement/stage variation testing/stiffness_summary.csv'

df=pd.read_csv(filename)

data_stage=pd.concat([df.loc[df['Status']=="before"],df.loc[df['Status']=="/"]],axis=0)
data_temp=df.loc[df['Status']!="/"]

data_stage=data_stage.groupby(by=['Date','Sample'])

plt.figure(figsize=(6, 4))

for name,group in data_stage:
    plt.plot(group['Somites'],group['Stiffness'],marker='o')

plt.show()
