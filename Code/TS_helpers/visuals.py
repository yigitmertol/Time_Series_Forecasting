import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

## SOME plotting code from notebooks
# TODO: Generalize and turn them into functions

fig, ax1 = plt.subplots()

# Create some mock data
t = volume.index
data1 = volume.values
data2 = (returns.values)

ax1.set_xlabel('Days')
ax1.set_ylabel('Volume (Currency)', color='lightblue')
ax1.fill_between(t, 0, data1, color='lightblue' )
ax1.tick_params(axis='y', labelcolor='lightblue')
ax1.tick_params(axis='x', rotation=60)
ax1.set_ylim(0, 500)

ax2 = ax1.twinx()


ax2.set_ylabel('Log Returns', color='grey')  # we already handled the x-label with ax1
ax2.bar(t, data2, color='grey')
ax2.tick_params(axis='y', labelcolor='grey')
ax2.set_ylim(-0.5, 0.5)


plt.xticks([x for x in volume.index if str(x)[-2:]=='01'],
           [str(x)[:7] for x in volume.index if str(x)[-2:]=='01'])

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()