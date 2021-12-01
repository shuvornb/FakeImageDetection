import os
import time
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import torch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import fakeimagedetection.edsr as edsr
import imghdr
from tensorflow.keras.utils import to_categorical
import numpy as np
from keras.preprocessing import image
from keras.applications.xception import preprocess_input


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


def get_upsampled(data_path):
    src_path = data_path
    file_names = []
    train_data = []

    upsampled_path = os.path.join('/'.join(src_path.split('/')[0:-1]), 'upsampled')
    os.mkdir(upsampled_path)
    model = edsr.load_edsr(device="cuda", scale=4)
    model.eval()
    counter = 0
    print('Upsampling images...')
    for e in os.listdir(src_path):
        new_path = os.path.join(upsampled_path, e)
        os.mkdir(new_path)
        path = os.path.join(src_path, e)
        for x in os.listdir(path):
            counter += 1
            img = cv2.imread(os.path.join(path, x))

            low_batch = torch.tensor(img).float().unsqueeze(0)
            low_batch = torch.permute(low_batch, (0, 3, 1, 2))
            low_batch = low_batch.to("cuda")
            with torch.no_grad():
                sr_images = model(low_batch)
                sr_images = sr_images.to("cpu").int()
            sr_images = torch.permute(sr_images, (0, 2, 3, 1))
            sr_images = sr_images.numpy()
            sr_image = sr_images[0]
            cv2.imwrite(os.path.join(new_path, x), sr_image)

            low_batch = low_batch.to("cpu")
            del low_batch
            del img
            del sr_image
    model.cpu()
    del model

    print('Total upsampled images: ', counter)
    return upsampled_path


def generate_from_paths_and_labels(input_paths, labels, batch_size, input_size=(256, 256)):
    num_samples = len(input_paths)
    while 1:
        perm = np.random.permutation(num_samples)
        input_paths = input_paths[perm]
        labels = labels[perm]
        for i in range(0, num_samples, batch_size):
            inputs = list(map(
                lambda x: image.load_img(x, target_size=input_size),
                input_paths[i:i+batch_size]
            ))
            inputs = np.array(list(map(
                lambda x: image.img_to_array(x),
                inputs
            )))
            inputs = preprocess_input(inputs)
            yield inputs, labels[i:i + batch_size]


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


def batch_data_xception(dataset_root):
    classes = ["fake", "real"]
    num_classes = len(classes)

    # make input_paths and labels
    input_paths, labels = [], []
    for class_name in os.listdir(dataset_root):
        class_root = os.path.join(dataset_root, class_name)
        class_id = classes.index(class_name)
        for path in os.listdir(class_root):
            path = os.path.join(class_root, path)
            if imghdr.what(path) is None:
                # this is not an image file
                continue
            input_paths.append(path)
            labels.append(class_id)

    # convert to one-hot-vector format
    labels = to_categorical(labels, num_classes=num_classes)

    # convert to numpy array
    input_paths = np.array(input_paths)

    # shuffle dataset
    perm = np.random.permutation(len(input_paths))
    labels = labels[perm]
    input_paths = input_paths[perm]
    return input_paths, labels


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
    #plt.show()


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
    pass
