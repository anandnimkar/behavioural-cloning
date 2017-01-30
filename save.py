import argparse
import keras

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model')
    parser.add_argument('--model', default=None, help='Model to load')
    args = parser.parse_args()
    
    # load previous weights if specified
    if args.model:
        model = keras.models.load_model(args.model)
        
        target = args.model.split('/')[-1].replace('.h5', '')
        
        print('Saving weights and model')
    
        with open('./saved/{}.json'.format(target), 'w') as f:
            f.write(model.to_json())

        model.save_weights('./saved/{}.h5'.format(target), True)
        
    