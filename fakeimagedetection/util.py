from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import pandas as pd
import os


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


def plot_mutliple_lines(data, title, x_label, y_label, legend):
    for i in range(len(data)):
        plt.plot(data[i])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legend)
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
