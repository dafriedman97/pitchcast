import pandas as pd
import numpy as np

def add_pitcher_pitch_rates(df, min_pitches=100):
    """For each pitch, add pitcher's frequency of each pitch-type from all previous pitches"""
    df['nth_pitch'] = df.groupby('pitcher_id').cumcount() + 1
    pitch_types = df['pitch_type'].unique()
    pitch_type_dummies = pd.get_dummies(df[['pitcher_id', 'pitch_type']], columns=['pitch_type'], prefix='', prefix_sep='')
    pitch_type_counts = pitch_type_dummies.groupby('pitcher_id').cumsum()
    pitch_type_rates = pitch_type_counts[pitch_types].divide(df['nth_pitch'], axis=0)
    pitch_type_rates.columns = [x+"_rate" for x in pitch_type_rates.columns]
    df = df.join(pitch_type_rates)
    df.loc[df['nth_pitch'] < min_pitches, list(pitch_type_rates.columns)] = np.nan
    return df
