import matplotlib.pyplot as plt
import numpy as np

with plt.xkcd():
    # Based on "Stove Ownership" from XKCD by Randall Munroe
    # https://xkcd.com/418/

    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([0, 100])

    
    experiment_x=np.random.randint(1, 100, size=(1,50))
    experiment_y=np.random.randint(30, 70, size=(1,50))
    model=60*np.ones(100)-0.5*np.arange(100)

    ax.plot(model,color='g',label='model')
    ax.scatter(experiment_x,experiment_y,label='experiment')

    ax.legend()

    plt.show()