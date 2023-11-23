# Created on Mon Aug 21 15:15:00 2023 - by Ahmed Yasser Eita
# from "https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py"

import matplotlib.pyplot as plt
import numpy as np

# x axis
variants = ("ModelNet10_12v", "ModelNet10_20v", "ModelNet40_12v", "ModelNet40_20v")

# bar values
accs = {
    'VAE': (77.0, 95.9, 67.0, 50.36),
    'VAeViT': (98.12, 99.67, 96.88, 98.02),
}

x = np.arange(len(variants))    # define the labels' array size
width = 0.25                    # the width of the bars
pos_mult = 0.5                  # position multiplier -> adjusts the labels' position

# PLOT
fig, ax = plt.subplots(layout='constrained')

for models, values in accs.items():
    offset = width * pos_mult
    bars = ax.bar(x + offset, values, width, label=models) # create the bars
    ax.bar_label(bars, padding=2)
    pos_mult += 1

# Define the axis names, title, legend, and some plotting adjustments
ax.set_ylabel('Accuracy')
ax.set_xlabel('Dataset')
# ax.set_title('')
ax.set_xticks(x + width, variants)
ax.legend(loc='lower center', ncols=3)
ax.set_ylim(0, 110)

# save the plot as a pdf file
plt.savefig('Bar_plot_VAE_VAeViT.pdf', bbox_inches='tight')
plt.show()
