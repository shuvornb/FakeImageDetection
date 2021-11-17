from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import pandas as pd
import os
import fakeimagedetection.edsr as edsr
import torch
import cv2
import numpy as np
def upsample_image(image, scale=4):
    model = edsr.load_edsr(device="cuda", scale=scale)
    image = image.to("cuda")
    model.eval()
    with torch.no_grad():
        sr_image = model(image)
        sr_image = sr_image.to("cpu")
    model.cpu()
    del model
    return sr_image

def batch_data_v2(data_path,
                       batch_size,
                       scale=4,
                       shuffle=True):
    src_path = data_path
    train_data = []
    train_label = []
    lab = 1
    for e in os.listdir(src_path):
        path = os.path.join(src_path,e)
        for x in os.listdir(path):
            img = cv2.imread(os.path.join(path,x))
            train_data.append(img)
            train_label.append(lab)
        lab -= 1
    if scale > 1:
        model = edsr.load_edsr(device="cuda", scale=scale)
        model.eval()
        for start in range(0, len(train_data), batch_size):
            low_batch = train_data[start:start+batch_size]
            
            low_batch = torch.tensor(low_batch).float()
            low_batch = torch.permute(low_batch, (0,3,1,2))
            low_batch = low_batch.to("cuda")
            with torch.no_grad():
                sr_images = model(low_batch)
                sr_images = sr_images.to("cpu").to(dtype = torch.int8)
            sr_images = torch.permute(sr_images, (0,2,3,1))
            sr_images = sr_images.numpy()
            train_data[start:start+batch_size] = sr_images
        model.cpu()
        del model
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    print(train_data.shape)
    data_generator = ImageDataGenerator(rescale=1.0 / 255)
    batches = data_generator.flow(train_data, train_label,
                                  batch_size=batch_size,
                                  shuffle=shuffle)
    return batches

def batch_data(
    folder_path,
    target_size,
    batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
):
    data_generator = ImageDataGenerator(rescale=1.0 / 255)
    batches = data_generator.flow_from_directory(
        folder_path,
        target_size=target_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode=class_mode,
        shuffle=shuffle,
    )
    return batches


def plot_mutliple_lines(data, title, x_label, y_label, legend, save=False, model_name="", data_name=""):
    for i in range(len(data)):
        plt.plot(data[i])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legend)
    if save:
        file_name = generate_date_name(model_name + "_" + data_name + "_" + y_label)
        current_folder = os.path.dirname(os.path.realpath(__file__))
        file_path = current_folder + "/results/" + file_name + ".png"
        plt.savefig(file_path)
    plt.show()



def generate_date_name(name):
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    return name + "_" + date


def get_gpu_details():
    print(tf.config.list_physical_devices("GPU"))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


def stats_to_csv(dictionary, model_name, data_name):
    data_frame = pd.DataFrame(dictionary)
    file_name = generate_date_name(model_name + "_" + data_name)
    current_folder = os.path.dirname(os.path.realpath(__file__))
    file = open(current_folder + "/results/" + file_name + ".csv", "w")
    data_frame.to_csv(file)
    file.close()
    
if __name__ == "__main__":
    batch_data_v2('sample_data/trial/train', (256,256), 1)
