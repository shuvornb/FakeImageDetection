from tensorflow import keras


def test_model(trained_model_path, data):
    trained_mdoel = keras.models.load_model(trained_model_path)
    test_score = trained_mdoel.evaluate(data, verbose=1)
    return test_score
