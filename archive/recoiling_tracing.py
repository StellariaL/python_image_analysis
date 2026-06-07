import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

raw_folder="D:\\Xiong Lab\\recoiling tracing\\GM6001\\"
output_path="D:\\Xiong Lab\\recoiling tracing\\GM6001\\summary_2term.csv"


timepoints={'5':0,
            '15':1,
            '30':2,
            '60':3}

# ===== models =====
def exp0(x,A,k):
    return A*np.exp(-k*x)

def exp1(x, A, k, C):
    return A * np.exp(-k * x) + C

def exp2(x, A1, k1, A2, k2, C):
    return A1 * np.exp(-k1 * x) + A2 * np.exp(-k2 * x) + C

def exp3(x, A1, k1, A2, k2):
    return A1 * np.exp(-k1 * x) + A2 * np.exp(-k2 * x)

colors = plt.cm.gray(np.linspace(0.7, 0.3, len(timepoints)))

fig,ax=plt.subplots()

fit_list=[]
for filename in os.listdir(raw_folder):
    if filename.endswith('.xlsx'):
        namesplit=filename.split('-')
        holdtime=namesplit[0]
        date=namesplit[1]
        sample=namesplit[2]
        '''
        if date!='251116':
            continue
        '''
        if 'GM' in sample:
            color='r'
        else:
            color='b'
        print("processing "+filename)
        filepath=raw_folder+filename
        raw = pd.read_excel(filepath)
        raw['diff']=raw['x']-raw['xref']
        init=raw.loc[0,['diff']].to_numpy()
        raw['displacement']=init-raw['diff']
        max=raw.loc[1,['displacement']].to_numpy()
        raw['proportion']=raw['displacement']/max
        raw=raw.dropna()
        y_data=raw.iloc[1:, raw.columns.get_loc('displacement')].to_numpy()
        x_data=raw.iloc[1:, raw.columns.get_loc('time')].to_numpy()
        ax.plot(x_data,y_data,marker='o',mec=None, color=color,
                 label=sample)
        p0_0=[np.max(y_data),0.5]
        p0_1 = [np.max(y_data), 0.5, np.min(y_data)]
        p0_2 = [np.max(y_data)/2, 0.5, np.max(y_data)/2, 0.1, np.min(y_data)]  # initial guess
        p0_3=[np.max(y_data)/2, 0.5, np.max(y_data)/2, 0.1]
        try:
            '''
            params, _ = curve_fit(exp2, x_data, y_data, p0=p0_2)
            y_fit = exp2(x_data, *params)
            '''
            params, _ = curve_fit(exp3, x_data, y_data, p0=p0_3)
            y_fit = exp3(x_data, *params)
            resid = y_data - y_fit
            ss_res = np.sum(resid**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r2 = 1 - ss_res / ss_tot
            row={'date':date,
                 'sample':sample,
                 'holdtime':holdtime,
                 'A1':params[0],
                 'k1':params[1],
                 'A2':params[2],
                 'k2':params[3],
                 #'C':params[2],
                 'R2':r2}
            fit_list.append(row)
            ax.plot(x_data, y_fit,color=colors[timepoints[holdtime]])
        except RuntimeError as e:
            print(filename+" fit failed.")
ax.legend()
plt.show()


fit_result = pd.DataFrame(fit_list)
fit_result.to_csv(output_path)
