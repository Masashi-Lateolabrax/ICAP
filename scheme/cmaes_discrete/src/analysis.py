import numpy as np
import matplotlib.pyplot as plt

import framework

from loss import Loss
from settings import Settings


def plot_loss(settings: Settings, dump: framework.Dump, file_path):
    loss = Loss()

    loss_r = np.zeros(len(dump.robot_pos))
    loss_n = np.zeros(loss_r.shape)
    nest_pos = np.array(settings.Nest.POSITION)

    iter_ = zip(dump.robot_pos, dump.food_pos)
    for i, (robot_pos, food_pos) in enumerate(iter_):
        loss_r[i] = loss.calc_r_loss(robot_pos, food_pos)
        loss_n[i] = loss.calc_n_loss(nest_pos, food_pos)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()

    loss_line_r = ax1.plot(loss_r, label='robot loss', color='blue')
    loss_line_n = ax2.plot(loss_n, label='nest loss', color='orange')

    ax1.set_xlabel('iteration')
    ax1.set_ylabel('robot loss')
    ax2.set_ylabel('nest loss')
    ax1.set_title('Loss')

    fig.legend(handles=[loss_line_r[0], loss_line_n[0]], loc='lower center')

    fig.savefig(file_path)
