import pandas as pd
import numpy as np

def get_pitcher_level_info(df, min_pitches=100):
    """Store pitcher's rates and handedness for streamlit app"""
    df_last = df.groupby('pitcher_name').last()
    df_last = df_last.loc[df_last['nth_season_pitch'] >= min_pitches] # restrict to pitchers with >= min_pitches
    pitcher_level_info = df_last[
        ['pitcher_lefty', 'fastball_rate', 'sinker_rate', 'slider_rate', 'changeup_rate', 'knuckle_curve_rate', 'curveball_rate', 'cutter_rate', 'splitter_rate', 'other_rate']
    ]
    return pitcher_level_info
