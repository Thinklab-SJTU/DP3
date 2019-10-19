import numpy as np
import matplotlib.pyplot as plt

f1_means = (49.5, 61.21, 60.88, 61.42, 62.08)
mae_means = (5.43, 3.737, 3.123, 3.532, 3.007)

ind = np.arange(len(mae_means))  # the x locations for the groups
width = 0.7  # the width of the bars
color = 'gray'

fig, ax = plt.subplots()
rects = ax.bar(ind, mae_means, width, color=color, label='MAE')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(ind)
ax.set_xticklabels(('Hawkes', 'ERPP', 'ERPP+In', 'AERPP', 'AERPP+In'))
ax.legend()


def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


autolabel(rects, "center")

plt.show()