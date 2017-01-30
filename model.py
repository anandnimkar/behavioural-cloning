import datetime
import argparse
import numpy as np
import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Lambda, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from preprocess import preprocess_train, preprocess_test


def load_data(data_folders):
    dfs = []
    for d in data_folders:
        df = pd.read_csv(d + 'driving_log.csv') 
        df.center = d + df.center.str.strip()
        df.left = d + df.left.str.strip()
        df.right = d + df.right.str.strip()
        dfs.append(df.copy())
    return pd.concat(dfs, ignore_index=True)

def shift_steering(df, shift_value=0.25):
    center = df[['center', 'steering']].rename(columns={'center': 'image_path'})
    left = df[['left', 'steering']].rename(columns={'left': 'image_path'})
    right = df[['right', 'steering']].rename(columns={'right': 'image_path'})
    left['steering'] = np.minimum(left['steering'] + shift_value, 1.)
    right['steering'] = np.maximum(right['steering'] - shift_value, -1.)
    return pd.concat((center, left, right), ignore_index=True)

def generate(df, preprocess_f, batch_size=128):
    X, y = [], []
    while 1:
        df = shuffle(df)
        for idx, row in df.iterrows():
            image = cv2.imread(row['image_path'])
            image, steering = preprocess_f(image, row['steering'])
            image = image[None, :, :, :]
            X.append(image)
            y.append([steering])
            if len(X) >= batch_size:
                yield np.vstack(X), np.array(y)
                X, y = [], []

def get_model(im_shape=(66, 200, 3)):
    """
    Implementation of the nvidia model specified here: 
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    
    Concepts from the following model were also implemented:
    https://github.com/commaai/research/blob/master/train_steering_model.py
    """ 
    model = Sequential()
    
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=im_shape, output_shape=im_shape))
    
    model.add(Convolution2D(25, 5, 5, subsample=(2, 2), border_mode='valid', init='glorot_uniform'))
    model.add(ELU())
    
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', init='glorot_uniform'))
    model.add(ELU())

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', init='glorot_uniform'))
    model.add(ELU())
 
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', init='glorot_uniform'))
    model.add(ELU())
    
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', init='glorot_uniform'))
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add(Flatten())
    
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Dense(10))
    model.add(ELU())
    
    model.add(Dense(1))
    
    model.compile(optimizer="adam", loss="mse")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model')
    parser.add_argument('--data', nargs='+', default=['./data/sim_data/','./data/udacity_data/'], 
                        help='Directories from which to load driving data')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('--valid-split', type=float, default=0.25, help='Portion of total dataset to split for validation set')
    parser.add_argument('--train-size', type=int, default=40064, help='Number of training samples per epoch.')
    parser.add_argument('--valid-size', type=int, default=8192, help='Number of validation samples per epoch.')
    parser.add_argument('--load-h5', default=None, help='Weights to load')
    args = parser.parse_args()
    
    df = load_data(args.data)
    df = shift_steering(df)
    df_train, df_valid = train_test_split(df, test_size=args.valid_split)
    
    print('{} training examples'.format(df_train.shape[0]))
    print('{} validation examples'.format(df_valid.shape[0]))
    
    model = get_model()
    model_name = 'model_' + datetime.datetime.now().strftime('%Y-&m-%dT%H-%M-%S')
    print('Model name: ' + model_name)
    
    # load previous weights if specified
    if args.load_h5:
        model.load_weights(args.load_h5, by_name=True)
    
    # callbacks
    stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(filepath='checkpoints/' + model_name + '_epoch:{epoch:02d}-val_loss:{val_loss:.4f}.h5',
                                 monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)                    
    # generators
    train_gen = generate(df_train, preprocess_f=preprocess_train)
    valid_gen = generate(df_valid, preprocess_f=preprocess_test)
    
    
    # model training and evaluation against validation set
    model.fit_generator(train_gen, 
                        samples_per_epoch=args.train_size, 
                        nb_epoch=args.epochs,
                        validation_data=valid_gen, 
                        nb_val_samples=args.valid_size,
                        callbacks=[stop, checkpoint])
    
    # saving model
    print('Saving weights and model')
    
    with open('./saved/{}.json'.format(model_name), 'w') as f:
        f.write(model.to_json())
        
    model.save_weights('./saved/{}.h5'.format(model_name), True)
    
    
    
    
    
