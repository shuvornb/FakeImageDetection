import math

from keras import callbacks
from keras.losses import categorical_crossentropy
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
import fakeimagedetection.util as util


def train_model(model, train_batches, valid_batches, config, verbose=1):
    """
    Trains a model

    Parameters
    ----------
    model : keras model
        model object.
    train_batches : data iterator
        DEStraining setCRIPTION.
    valid_batches : data iterator
        validation set.
    config : dict
        parameterrs.
    verbose : bool, optional
        print details or not. The default is 1.

    Returns
    -------
    model : trained model
        model object.
    history : dict
        distory of training.

    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config["learning_rate"]),
        loss=config["loss"],
        metrics=config["metrics"],
    )
    early_stopping = callbacks.EarlyStopping(
        monitor=config["monitor"],
        min_delta=config["min_delta"],
        patience=config["patience"],
        verbose=verbose,
        mode="auto",
    )
    history = model.fit(
        train_batches,
        steps_per_epoch=config["steps_per_epoch"],
        epochs=config["epochs"],
        validation_data=valid_batches,
        validation_steps=config["validation_steps"],
        callbacks=[early_stopping],
    )
    return model, history


def train_xception_model(
    model, dataset_root_train, dataset_root_valid, config, verbose=1
):
    """
    Trains an Xception model

    Parameters
    ----------
    model : keras model
        model object.
    train_batches : data iterator
        DEStraining setCRIPTION.
    valid_batches : data iterator
        validation set.
    config : dict
        parameterrs.
    verbose : bool, optional
        print details or not. The default is 1.

    Returns
    -------
    model : trained Xceptionmodel
        model object.
    history : dict
        distory of training.

    """
    train_input_paths, train_labels = util.batch_data_xception(dataset_root_train)
    val_input_paths, val_labels = util.batch_data_xception(dataset_root_valid)
    print("Training on %d images and labels" % (len(train_input_paths)))
    print("Validation on %d images and labels" % (len(val_input_paths)))

    # compile model
    model.compile(
        loss=categorical_crossentropy, optimizer=Adam(lr=5e-3), metrics=["accuracy"]
    )

    early_stopping = callbacks.EarlyStopping(
        monitor=config["monitor"],
        min_delta=config["min_delta"],
        patience=config["patience"],
        verbose=verbose,
        mode="auto",
    )

    # dataset
    history = model.fit_generator(
        generator=util.generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=config["train_batch_size"],
        ),
        steps_per_epoch=math.ceil(len(train_input_paths) / config["train_batch_size"]),
        epochs=config["epochs"],
        validation_data=util.generate_from_paths_and_labels(
            input_paths=val_input_paths,
            labels=val_labels,
            batch_size=config["train_batch_size"],
        ),
        validation_steps=math.ceil(len(val_input_paths) / config["train_batch_size"]),
        verbose=1,
        callbacks=[early_stopping],
    )
    return model, history


if __name__ == "__main__":
    pass
