from keras import callbacks
from tensorflow.keras import optimizers


def train_model(model, train_batches, valid_batches, config, verbose=1):
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
        epochs=config["epochs"],
        validation_data=valid_batches,
        callbacks=[early_stopping],
    )
    return model, history


if __name__ == "__main__":
    pass
