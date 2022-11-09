import argparse
import os
import json
import pandas as pd
import numpy as np
from tensorflow import keras
import keras_tuner
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hypermodel import CustomHyperModel
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

## Load data ##
def load_data(seasons):
    """Concat training data from provided training seasons"""
    season_dfs = list()
    for season in seasons:
        season_path = os.path.join(root_dir, f"data/{season}.csv")
        if not os.path.exists(season_path):
            raise Exception(f"{season} dataframe not yet built. Build first with `python pipeline/main.py -s {season}`")
        season_dfs.append(pd.read_csv(season_path))
    df = pd.concat(season_dfs).reset_index(drop=True)
    return df
 
## Prep data for modeling ##
def create_train_and_test_data(df, simple=False):
    """Create train/test data"""
    df, dummy_columns = get_dummies(df, simple=simple)
    features = get_features(dummy_columns, simple=simple)
    X = df.loc[df['raw_pitch_type'] != "other"].dropna(subset=features)[features]
    y = df.loc[df['raw_pitch_type'] != "other"].dropna(subset=features)['pitch_type']
    y = pd.get_dummies(y)
    y = y[['fastball', 'curveball', 'sinker', 'cutter', 'changeup', 'slider', 'splitter', 'knuckle_curve', 'other']] # reorder
    X_train_numpy, X_val_numpy, y_train_numpy, y_val_numpy = scale_and_split(X, y, simple)
    return X_train_numpy, X_val_numpy, y_train_numpy, y_val_numpy

def get_dummies(df, simple=False):
    """Create dummies for dataframe"""
    if simple:
        dummy_columns = ['count']
        prefixes = ['count']
    else:
        dummy_columns = ['pitch_type_lag_1', 'pitch_type_lag_2']
        prefixes = ['count', 'lag_1', 'lag_2']
    dummy_df = pd.get_dummies(df[dummy_columns], prefix=prefixes)
    return df.join(dummy_df), dummy_df.columns

def get_features(dummy_columns, simple=False):
    """Collect features for training"""
    # Plain/direct features
    game_state = ['inning', 'top', 'outs']
    runners = ['runner_1', 'runner_2', 'runner_3']
    scores = ['home_score', 'away_score']
    pitch_counts = ['pitch_count', 'inning_pitch_count', 'ab_pitch_count']
    hands = ['pitcher_lefty', 'batter_lefty']
    features = game_state + runners + scores + pitch_counts + hands

    # Pitch type
    pitch_types = ['fastball', 'curveball', 'sinker', 'cutter', 'changeup', 'slider', 'splitter', 'knuckle_curve', 'other']
    pitch_type_rates = [pitch_type + "_rate" for pitch_type in pitch_types]
    features.extend(pitch_type_rates)

    # Dummy columns
    features.extend(dummy_columns)

    # At-Bat pitch type counts (if full model)
    if not simple:
        ab_pitch_type_counts = [f"ab_{pitch_type}_count" for pitch_type in pitch_types]
        features.extend(ab_pitch_type_counts)

    # Return
    return features 
 
def scale_and_split(X, y, simple=False):
    """Split train/test and scale off of train"""
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    X_train_numpy = X_train.astype(float).to_numpy()
    y_train_numpy = y_train.astype(float).to_numpy()
    X_val_numpy = X_val.astype(float).to_numpy()
    y_val_numpy = y_val.astype(float).to_numpy()
    if not simple: # simple model has no scaling so input can be used directly
        ss = StandardScaler()
        X_train_numpy = ss.fit_transform(X_train_numpy)
        X_val_numpy = ss.transform(X_val_numpy)
    return X_train_numpy, X_val_numpy, y_train_numpy, y_val_numpy

## Tune and return model ##
def tune_model(X_train_numpy, X_val_numpy, y_train_numpy, y_val_numpy):
    """Tune model with keras_tuner and return best model"""
    input_shape = X_train_numpy.shape[1]
    output_shape = y_train_numpy.shape[1]
    with open(os.path.join(root_dir, "model/config.json"), "r") as f:
        config = json.load(f)
    tuner = keras_tuner.RandomSearch(
        hypermodel=CustomHyperModel(input_shape, output_shape),
        objective='val_loss',
        max_trials=50,
        executions_per_trial=1,
        directory=os.path.join(root_dir, "model"),
        project_name='keras_tuner',
        overwrite=True,
    )
    tuner.search(
        X_train_numpy,
        y_train_numpy,
        validation_data=(X_val_numpy, y_val_numpy),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=[keras.callbacks.EarlyStopping(patience=config['patience'])]
    )
    model = tuner.get_best_models()[0]
    return model

if __name__ == "__main__":    
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--first_training_season", type=int)
    parser.add_argument("-l", "--last_training_season", type=int)
    parser.add_argument("-n", "--model_name", type=str, default="fit_model")
    parser.add_argument('-s', '--simple', action='store_true', help="build simple model (for streamlit app)")
    args = parser.parse_args()
    first_training_season = args.first_training_season
    last_training_season = args.last_training_season
    model_name = args.model_name
    simple_model = args.simple
    if simple_model:
        model_name += "_simple"
    saved_models_dir = os.path.join(root_dir, "model/saved_models")
    if not os.path.exists(saved_models_dir):
        os.mkdir(saved_models_dir)
    
    # # Load data
    # training_seasons = np.arange(first_training_season, last_training_season+1)
    # df = load_data(training_seasons)
 
    # # Clean/transform/train test split
    # X_train_numpy, X_val_numpy, y_train_numpy, y_val_numpy = create_train_and_test_data(df, simple=simple_model)
    
    # # Tune
    # model = tune_model(X_train_numpy, X_val_numpy, y_train_numpy, y_val_numpy)

    # # Write out
    # model.save(os.path.join(saved_models_dir, f"{model_name}.h5"))
