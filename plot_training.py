import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def my_plt(path_file):
    f = open(path_file, "r")
    lines = f.readlines()
    class_name = lines[4].split(' ')[2].replace('/', '_')
    train_losses = []
    val_losses = []
    epochs = []
    for line in lines[6:]:
        train_losses.append(float(line.split('***')[1].split(':')[1]))
        val_losses.append(float(line.split('***')[2].split(':')[1]))
        epochs.append(int(line.split('***')[0].split(':')[1]))
    plt.plot(epochs, train_losses, label='Training loss')
    plt.plot(epochs, val_losses, label='Validation loss')
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.title(class_name)
    plt.savefig('{}.png'.format(class_name))




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plots tarining and validation objective function.")
    parser.add_argument("-p", "--path_file", default = '.{}'.format(os.sep), type = str, help="Path to log file")

    args = parser.parse_args()
    my_plt(args.path_file)
