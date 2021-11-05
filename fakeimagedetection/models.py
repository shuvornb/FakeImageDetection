from tensorflow.keras import layers, models
import tensorflow as tf


def meso4_model():
    model = models.Sequential()

    model.add(layers.Conv2D(8, (3, 3), input_shape=(256, 256, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    model.add(layers.Conv2D(8, (5, 5), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    model.add(layers.Conv2D(16, (5, 5), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    model.add(layers.Conv2D(16, (5, 5), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((4, 4), padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def xception():
    base_model = tf.keras.applications.xception.Xception(include_top=False,
                                                         weights='imagenet',
                                                         input_shape=(256, 256, 3))

    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=2000,  # was 256
                     activation='relu',
                     # kernel_initializer=initializer
                     )(x)

    x = layers.Dropout(0.4)(x)  # was 0.4
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=1000,  # was 128
                     activation='relu',
                     # kernel_initializer=initializer
                     )(x)

    x = layers.Dropout(0.4)(x)  # was 0.4
    x = layers.Dense(units=1,  # was 10
                     activation='sigmoid',
                     # kernel_initializer=initializer
                     )(x)

    # use for VGG
    # x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dense(4096, activation='relu')(x)
    # x = layers.Dense(10, activation='softmax')(x)
    pretrained = models.Model(inputs=base_model.input, outputs=x)
    pretrained.summary()
    return pretrained


def vgg():
    base_model = tf.keras.applications.vgg19.VGG19(input_shape=(256, 256, 3),
                                                   include_top=False,
                                                   weights='imagenet')

    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=2000,  # was 256
                     activation='relu',
                     # kernel_initializer=initializer
                     )(x)

    x = layers.Dropout(0.4)(x)  # was 0.4
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=1000,  # was 128
                     activation='relu',
                     # kernel_initializer=initializer
                     )(x)

    x = layers.Dropout(0.4)(x)  # was 0.4
    x = layers.Dense(units=1,  # was 10
                     activation='sigmoid',
                     # kernel_initializer=initializer
                     )(x)

    # use for VGG
    # x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dense(4096, activation='relu')(x)
    # x = layers.Dense(10, activation='softmax')(x)
    pretrained = models.Model(inputs=base_model.input, outputs=x)
    pretrained.summary()
    return pretrained


def resnet():
    base_model = tf.keras.applications.resnet.ResNet50(include_top=False,
                                                       input_shape=(256, 256, 3),
                                                       weights='imagenet')

    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=2000,  # was 256
                     activation='relu',
                     # kernel_initializer=initializer
                     )(x)

    x = layers.Dropout(0.4)(x)  # was 0.4
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=1000,  # was 128
                     activation='relu',
                     # kernel_initializer=initializer
                     )(x)

    x = layers.Dropout(0.4)(x)  # was 0.4
    x = layers.Dense(units=1,  # was 10
                     activation='sigmoid',
                     # kernel_initializer=initializer
                     )(x)

    # use for VGG
    # x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dense(4096, activation='relu')(x)
    # x = layers.Dense(10, activation='softmax')(x)
    pretrained = models.Model(inputs=base_model.input, outputs=x)
    pretrained.summary()
    return pretrained

def load_model(model_name):
    if model_name.lower() == 'meso4':
        model = meso4_model()
    elif model_name.lower() == 'xception':
        model = xception()
    elif model_name.lower() == 'resnet':
        model = resnet()
    elif model_name.lower() == 'vgg':
        model = vgg()
    else:
        print('Unknown model. Terminating program')
        return False
    return model