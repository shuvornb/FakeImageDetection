from tensorflow import keras


def test_model(trained_model_path, data, step):
    """
    Test model 

    Parameters
    ----------
    trained_model_path : str
        saved model.
    data : data iterator
        test data.
    step : int
        total testing steps.

    Returns
    -------
    test_score : list
        test scores.

    """
    trained_model = keras.models.load_model(trained_model_path)
    test_score = trained_model.evaluate_generator(data, verbose=1, steps=step)
    return test_score
