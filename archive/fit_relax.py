import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

raw_folder="D:\\Xiong Lab\\force-displacement\\holding\\"
output_path="D:\\Xiong Lab\\force-displacement\\holding\\fit_results_microsurgery_filtered.csv"

pattern = re.compile(r'([0-9]+)-r([A-Z]+)(\d+)_relax.csv')


# ===== models =====
def exp1(x, A, k, C):
    return A * np.exp(-k * x) + C

fig,ax=plt.subplots()


fit_list=[]
for filename in os.listdir(raw_folder):
    match = pattern.match(filename)
    if match:
        date = match.group(1)
        treatment = match.group(2)
        if treatment=='PD':
            color='red'
        else:
            color='blue'
        no=match.group(3)
        print("processing "+filename)
        filepath=raw_folder+filename
        raw = pd.read_csv(filepath)
        time=raw['time'].to_numpy()
        force=raw['force'].to_numpy()
        ax.plot(time,force,marker='o',
                markeredgecolor='black',markerfacecolor=color,
                label=date+'-'+treatment+no)
        p0 = [np.max(force), 0.001, np.min(force)]
        try:
            params, _ = curve_fit(exp1, time, force, p0=p0)
            y_fit = exp1(time, *params)
            resid = force - y_fit
            ss_res = np.sum(resid**2)
            ss_tot = np.sum((force - np.mean(force))**2)
            r2 = 1 - ss_res / ss_tot
            row={'date':date,
                 'treatment':treatment,
                 'sample':treatment+no,
                 'A':params[0],
                 'k':params[1],
                 'C':params[2],
                 'relaxed':params[0]/(params[0]+params[2]),
                 'R2':r2}
            fit_list.append(row)
            ax.plot(time, y_fit,color='black')
        except RuntimeError as e:
            print(filename+" fit failed.")
ax.legend()
plt.show()

fit_result = pd.DataFrame(fit_list)
fit_result.to_csv(output_path)