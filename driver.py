import toml
import sys
import click
import fakeimagedetection.models as models
import fakeimagedetection.util as util
import fakeimagedetection.train as train 

@click.command()
@click.option('--mode', default='train', help='Execution mode')
@click.option('--model_name', default='meso4', help='Model name')
@click.option('--data_path', default='fakeimagedetection/sample_data/deepfake', help='Data folder path')
def run(mode, model_name, data_path):
    if mode.lower() == 'train':
        config = toml.load('config.toml')
        
        # check if GPU available
        util.get_gpu_details()

        train_path = data_path + '/train'
        test_path = data_path + '/test'

        train_batches = util.batch_data(train_path, (config['image_height'], config['image_width']), config['train_batch_size'])
        test_batches = util.batch_data(test_path, (config['image_height'], config['image_width']), config['test_batch_size'])
            
        model = models.load_model(model_name)
        if not model:
            sys.exit()
            
        trained_model, history = train.train_model(model, train_batches, test_batches, config)
        trained_model.save('fakeimagedetection/saved_models/'+util.generate_date_name(model_name+'_'+data_path.split('/')[-1]))
        
        print(history.history)
        util.stats_to_csv(history.history, model_name, data_path.split('/')[-1])
        
        train_score = trained_model.evaluate(train_batches, verbose=1)
        print('Train accuracy:', train_score[1])

        valid_score = trained_model.evaluate(test_batches, verbose=1)
        print('Validation accuracy:', valid_score[1])

        plot_data = [history.history['accuracy'], history.history['val_accuracy']]
        util.plot_mutliple_lines(plot_data, 'model accuracy', 'epoch', 'accuracy', ['train', 'validation'])
        
        plot_data = [history.history['loss'], history.history['val_loss']]
        util.plot_mutliple_lines(plot_data, 'model loss', 'epoch', 'loss', ['train', 'validation'])

    elif model.lower() == 'test':
        pass
    elif mode.lower() == 'predict':
        pass
    else:
        print('Unknown mode. Terminating program')

if __name__ == "__main__":
    run()