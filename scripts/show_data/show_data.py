# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

class ShowData(object):
    def __init__(self):
        pass

    def show_loss_curve(self, input_data, save_name):
        ax = plt.subplot2grid((1,1), (0,0))
        ax.plot(range(len(input_data)), input_data, color="b")
        ax.set_xlabel("episode")
        ax.set_ylabel("loss")
        ax.grid()
        plt.savefig(save_name)

    def show_accuracy_curve(self, input_data, save_name):
        ax = plt.subplot2grid((1,1), (0,0))
        ax.plot(range(len(input_data)), input_data, color="b")
        ax.set_xlabel("episode")
        ax.set_ylabel("accuracy")
        ax.grid()
        plt.savefig(save_name)

    def show_accuracy_curves(self, input_data1, input_data2, save_name):
        ax = plt.subplot2grid((1,1), (0,0))
        ax.plot(range(len(input_data1)), input_data1, color="b")
        ax.plot(range(len(input_data2)), input_data2, color="r")
        ax.set_xlabel("episode")
        ax.set_ylabel("accuracy")
        ax.grid()
        plt.ylim([0,85])
        plt.savefig(save_name)