import matplotlib.pyplot as plt
import tensorflow as tf
from keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import models


if __name__ == '__main__':

    # check if GPU available
    print(tf.config.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # data path
    train_path = '/home/shuvornb/Desktop/Deepfake Detection/DeepFake-Detection-master/data/dataset1/train'
    test_path = '/home/shuvornb/Desktop/Deepfake Detection/DeepFake-Detection-master/data/dataset1/test'

    # define train data generator
    train_data_generator = ImageDataGenerator(
        rescale=1. / 255,
    )
    train_batches = train_data_generator.flow_from_directory(
        train_path,
        target_size=(256, 256),
        batch_size=128,
        color_mode='rgb',
        class_mode='binary'
    )

    # define test data generator
    test_data_generator = ImageDataGenerator(
        rescale=1. / 255
    )
    test_batches = test_data_generator.flow_from_directory(
        test_path,
        target_size=(256, 256),
        batch_size=32,
        color_mode='rgb',
        class_mode='binary',
        shuffle=True
    )

    # get model instance and compile it with optimizer and error function
    model = models.meso4_model()
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy'])
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=1,
        mode="auto"
    )

    # train the model
    history = model.fit(
        train_batches,
        steps_per_epoch=413,
        epochs=50,
        validation_data=test_batches,
        validation_steps=108,
        callbacks=[early_stopping]
    )

    train_score = model.evaluate(train_batches, verbose=1)
    print('Train accuracy:', train_score[1])

    valid_score = model.evaluate(test_batches, verbose=1)
    print('Validation accuracy:', valid_score[1])

    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # save the model
    model.save("saved_models/meso4")