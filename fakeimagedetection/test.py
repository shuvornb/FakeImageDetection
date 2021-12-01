from tensorflow import keras


def test_model(trained_model_path, data, step):
    trained_mdoel = keras.models.load_model(trained_model_path)
    test_score = trained_mdoel.evaluate_generator(data, verbose=1, steps=step)
    return test_score