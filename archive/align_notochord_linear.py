import os
import numpy as np
import pandas as pd
import scipy.odr as odr
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
#from statannotations.Annotator import Annotator


def linear_model(beta, x):
    return beta[0] + beta[1] * x

def significance_text(p):
    """Returns significance asterisks based on p-value."""
    if p < 0.001:
        return "***"  # Highly significant
    elif p < 0.01:
        return "**"   # Significant
    elif p < 0.05:
        return "*"    # Weakly significant
    else:
        return "n.s."  # Not significant

ref_length=500 # the point this many pixel away from the endpoint will be aligned to the x axis.
ref_ind=int(-ref_length-1)

treatment_info_path="E:\\Xiong Lab\\axis shape data\\treatment_info.csv"
raw_folder="E:\\Xiong Lab\\axis shape data\\"

treatment_info = pd.read_csv(treatment_info_path)

# Separate treatments
ctrl_info = treatment_info[treatment_info['treatment'] == '4ctrl']
pd_info = treatment_info[treatment_info['treatment'] == '4PD']
all_MAR=[]
all_RMSR=[]

# Initialize subplots
fig1, axes = plt.subplots(2,1, sharey=True)
axes[0].set_title('DMSO')
axes[1].set_title('PD173074')


for ax, info, title in zip(axes, [ctrl_info, pd_info], ['4ctrl', '4PD']):
    group_MAR=[]
    group_RMSR=[]
    for i, row in info.iterrows():
        filename = f"{row['date']}-{row['sample']}-day2.csv"
        filepath = os.path.join(raw_folder, filename)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        try:
            raw = pd.read_csv(filepath)
            x = raw['x'].to_numpy()
            y = raw['y'].to_numpy()
            x_aligned = x - x[-1]
            x_fitting=np.array(x_aligned[ref_ind:-1])
            y_fitting=np.array(y[ref_ind:-1])
            p=np.polyfit(x_fitting,y_fitting,1)
            model = odr.Model(linear_model)
            data = odr.Data(x_fitting,y_fitting)
            odrfit = odr.ODR(data, model, beta0=[p[1],p[0]])
            odr_result = odrfit.run()
            beta=odr_result.beta
            orthogonal_distances = np.sqrt(odr_result.delta**2 + odr_result.eps**2)
            MAR = np.mean(orthogonal_distances)
            RMSR=np.sqrt(np.mean(odr_result.delta**2 + odr_result.eps**2))
            group_MAR.append(MAR)
            group_RMSR.append(RMSR)
            y_aligned = y - beta[0]
            angle = -np.arctan2(-beta[1], -1)
            x_rotated = x_aligned * np.cos(angle) - y_aligned * np.sin(angle)
            y_rotated = x_aligned * np.sin(angle) + y_aligned * np.cos(angle)
            ax.plot(x_rotated, y_rotated)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    ax.set_aspect('equal')
    ax.set(xlim=(-100,1600),ylim=(-500,500))
    ax.set_xticks([])
    ax.set_yticks([])
    scalebar = ScaleBar(2.195749775, "um", length_fraction=0.2)
    ax.add_artist(scalebar)
    all_MAR.append(group_MAR)
    all_RMSR.append(group_RMSR)


plt.suptitle('Aligned Notochord Shapes')
plt.tight_layout()



fig2,axes2=plt.subplots()

sns.boxplot(all_RMSR,ax=axes2,width=.5,showcaps=False,fill=False,color='k')
sns.swarmplot(all_RMSR,ax=axes2,color='k')
axes2.set_xticklabels(["DMSO","PD173074"])
axes2.set_ylabel("Deviation")
axes2.spines['top'].set_visible(False)
axes2.spines['right'].set_visible(False)
u_stat, p_val = stats.mannwhitneyu(all_RMSR[0], all_RMSR[1], alternative="two-sided")
y_max=max(all_RMSR[0]+all_RMSR[1])
sig_annot = significance_text(p_val)
p_text = f"p = {p_val:.3f}"
axes2.plot([0, 1], [y_max*1.05, y_max*1.05], lw=1.5, c="black")
axes2.text(0.5, y_max*1.06, sig_annot, ha="center", fontsize=12, fontweight="bold")
axes2.text(0.5, y_max*1.11, p_text, ha="center", fontsize=12, fontweight="bold")

'''
sns.boxplot(all_RMSR,ax=axes2[1],width=.5,showcaps=False,fill=False,linecolor='k')
sns.swarmplot(all_RMSR,ax=axes2[1],color='k')
axes2[1].set_xticklabels(["DMSO","PD173074"])
u_stat, p_val = stats.mannwhitneyu(all_RMSR[0], all_RMSR[1], alternative="two-sided")
y_pos=max(all_RMSR[0]+all_RMSR[1])*1.1
p_text = f"p = {p_val:.3f}" if p_val >= 0.05 else "p < 0.05"
axes2[1].text(0.5, y_pos, p_text, ha="center", fontsize=12, fontweight="bold")
'''

plt.show()