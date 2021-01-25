"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping


import mlflow
import mlflow.keras
from mlflow import pyfunc
import os



warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))

    tags = {
        "usuario": "Anonymous"
    }
    
    mlflow.set_experiment("traffic_flow-saes")    
    with mlflow.start_run() as run:
        mlflow.set_tags(tags)
        mlflow.keras.autolog()
        
        model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
        #early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
        hist = model.fit(
            X_train, y_train,
            batch_size=config["batch"],
            epochs=config["epochs"],
            validation_split=0.05)

        model.save('model/' + name + '.h5')
        df = pd.DataFrame.from_dict(hist.history)
        df.to_csv('model/' + name + ' loss.csv', encoding='utf-8', index=False)
        mlflow.log_param("Run_id", run.info.run_id)
        #mlflow.log_artifact("./picture.png")        
        #mlflow.keras.log_model(keras_model=model, artifact_path="model")

            #PARTE PARA CONVERTIR A ONNX
            # load saved keras model and convert it to onnx model for loading as artifact in Mlflow
            #loaded_model = load_model('./model/gru.h5')
            #onnx_model = onnxmltools.convert_keras(loaded_model)        
            #mlflow.onnx.log_model(onnx_model = onnx_model, artifact_path="model")


def train_seas(models, X_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    
    
    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(input=p.input,
                                       output=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="saes",
        help="Model to train.")
    args = parser.parse_args()

    lag = 12
    config = {"batch": 256, "epochs": 50}
    file1 = 'data/train.csv'
    file2 = 'data/test.csv'
    X_train, y_train, _, _, _ = process_data(file1, file2, lag)

    if args.model == 'lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    if args.model == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_gru([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    if args.model == 'saes':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = model.get_saes([12, 400, 400, 400, 1])
        train_seas(m, X_train, y_train, args.model, config)


if __name__ == '__main__':    
    main(sys.argv)
