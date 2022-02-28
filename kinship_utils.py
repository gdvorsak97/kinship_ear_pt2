import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras_vggface.utils import preprocess_input
from tensorflow.python.keras.layers import Rescaling
from tensorflow.python.keras.models import Sequential

import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_loss(history, label, n):
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                 color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                 color=colors[n], label='Val ' + label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0, 1])  # originally from 0.8 to 1
        else:
            plt.ylim([0, 1])

        plt.legend()
    plt.show()


def visualize_crop(in_img, crp_img):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(in_img[:, :, :])
    axes[0].set_title('Original image')
    axes[1].imshow(crp_img[:, :, :])
    axes[1].set_title('Cropped input')
    fig.suptitle(f'Original and cropped input')
    fig.set_size_inches(9, 5, forward=True)
    plt.show()


def alignment(image, path, visualize=False, save=False):

    label_path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\bounding boxes alligment\\"
    delete_ist_path = label_path + "delete list.txt"
    filename = ""
    if "/" in path:
        family = path.split("/")[-3]
        filename = path.split("/")[-1]
        label_path += "labels " + str(family) + ".csv"
    elif "\\" in path:
        filename = path.split("\\")[-1]
        label_path = label_path + "all_labels.csv"
    label_file = pd.read_csv(label_path)
    bbox_data = label_file[label_file['file'] == filename]
    bbox = image[bbox_data['y1'].values[0]:bbox_data['y1'].values[0] + bbox_data['dy'].values[0],
           bbox_data['x1'].values[0]:bbox_data['x1'].values[0] + bbox_data['dx'].values[0]]
    if visualize:
        cv2.imshow("Detected", bbox)
        cv2.waitKey()
    if save:
        cv2.imwrite("example.png", bbox)
    return bbox


def crop_ears(img, region):
    if region == "left":
        img = img[:, 0:int(np.round(224 / 3))]
    elif region == "right":
        img = img[:, -int(np.round(224 / 3)):]
    elif region == "mid_vertical":
        img = img[:, int(np.round(224 / 6)):int(np.round(224 / 6)) + int(np.round(224 / 3))]
    elif region == "top":
        img = img[0:int(np.round(224 / 3)), :]
    elif region == "mid_horizontal":
        img = img[int(np.round(224 / 6)):int(np.round(224 / 6)) + int(np.round(224 / 3)), :]
    elif region == "bottom":
        img = img[-int(np.round(224 / 3)):, :]
    return img


# read images
def read_img(path):
    in_img = cv2.imread(path)
    in_img = in_img.astype('uint8')
    in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
    in_img = alignment(in_img, path)
    # in_img = cv2.resize(in_img, (224, 224))
    # in_img = crop_ears(in_img, "right")
    img = cv2.resize(in_img, (224, 224))
    # visualize_crop(in_img, img)
    return img


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

