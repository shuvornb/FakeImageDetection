import os

import toml
import sys
import click
import fakeimagedetection.models as models
import fakeimagedetection.util as util
import fakeimagedetection.train as train
import fakeimagedetection.test as test
import shutil
import math

@click.command()
@click.option('--mode', default='train', help='Execution mode')
@click.option('--model_name', default='meso4', help='Model name')
@click.option('--test_model_path', default=None, help='Trained model path')
@click.option('--data_path', default='fakeimagedetection/sample_data/deepfake', help='Data folder path')
@click.option('--upsample', default=False, help='Upsample images 4 times')
def run(mode, model_name, test_model_path, data_path, upsample):
    if mode.lower() == 'train':
        config = toml.load('config.toml')

        # check if GPU available
        util.get_gpu_details()

        if model_name == 'xception':
            model = models.load_model(model_name)
            if not model:
                sys.exit()
            trained_model, history = train.train_xception_model(model, data_path+"/train", data_path+"/test", config)
            trained_model.save('fakeimagedetection/saved_models/' + util.generate_date_name(
                model_name + '_' + data_path.split('/')[-1]))
            print(history.history)
            util.stats_to_csv(history.history, model_name, data_path.split('/')[-1])
            plot_data = [history.history['accuracy'], history.history['val_accuracy']]
            util.plot_mutliple_lines(plot_data, 'model accuracy', 'epoch', 'accuracy', ['train', 'validation'], True,
                                     model_name, data_path.split('/')[-1])
            plot_data = [history.history['loss'], history.history['val_loss']]
            util.plot_mutliple_lines(plot_data, 'model loss', 'epoch', 'loss', ['train', 'validation'], True,
                                     model_name, data_path.split('/')[-1])

        else:
            train_path = data_path + '/train'
            test_path = data_path + '/test'
            train_batches = util.batch_data(train_path, (config['image_height'], config['image_width']), config['train_batch_size'])
            test_batches = util.batch_data(test_path, (config['image_height'], config['image_width']), config['test_batch_size'])

            model = models.load_model(model_name)
            if not model:
                sys.exit()

            trained_model, history = train.train_model(model, train_batches, test_batches, config)
            trained_model.save('fakeimagedetection/saved_models/' + util.generate_date_name(model_name + '_' + data_path.split('/')[-1]))
            print(history.history)
            util.stats_to_csv(history.history, model_name, data_path.split('/')[-1])

            train_score = trained_model.evaluate(train_batches, verbose=1)
            print('Train accuracy:', train_score[1])
            valid_score = trained_model.evaluate(test_batches, verbose=1)
            print('Validation accuracy:', valid_score[1])

            plot_data = [history.history['accuracy'], history.history['val_accuracy']]
            util.plot_mutliple_lines(plot_data, 'model accuracy', 'epoch', 'accuracy', ['train', 'validation'], True, model_name, data_path.split('/')[-1])
            plot_data = [history.history['loss'], history.history['val_loss']]
            util.plot_mutliple_lines(plot_data, 'model loss', 'epoch', 'loss', ['train', 'validation'], True, model_name, data_path.split('/')[-1])

    elif mode.lower() == 'test':
        config = toml.load('config.toml')

        if not test_model_path:
            print('No test model path provided. Terminating program')

        test_path = data_path + '/test'
        if upsample:
            upsampled_path = util.get_upsampled(test_path)
            test_path = upsampled_path
        if 'xception' in test_model_path:
            input_paths, labels = util.batch_data_xception(test_path)
            print("Testing on %d images and labels" % (len(input_paths)))
            test_batches = util.generate_from_paths_and_labels(input_paths, labels, batch_size=config['test_batch_size'])
        else:
            test_batches = util.batch_data(test_path, (config['image_height'], config['image_width']),
                                           config['test_batch_size'])
        total_images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(test_path) for f in filenames if
                      os.path.splitext(f)[1] == '.png']
        test_score = test.test_model(test_model_path, test_batches, math.ceil(len(total_images)/config['test_batch_size']))
        print('Test accuracy:', test_score[1])

        if upsample:
            try:
                shutil.rmtree(test_path)
            except OSError as e:
                print("Error: %s : %s" % (test_path, e.strerror))

    elif mode.lower() == 'predict':
        pass
    else:
        print('Unknown mode. Terminating program')


if __name__ == "__main__":
    run()
