import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import numpy as np

root_path = './data'
p = Path(root_path)
files = p.glob('tot_metric*')
for file in files:
    data = numpy.load(file)
    train_loss = data['tot_train_loss']
    test_loss = data['tot_test_loss']
    train_time = data['tot_time']
    fig, ax = plt.subplots()
    ax.plot(train_loss, label='train loss')
    ax.plot(test_loss, label='test loss')
    ax.legend()
    plt.grid()

plt.show()