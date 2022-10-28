import pandas as pd
import numpy as np
from dataclasses import asdict

## Create raw dataframe from api values
def create_raw_data_frame(season_data):
    """Convert values returned from api into dataframe"""
    at_bat_dfs = list()
    for game_id, at_bats in season_data.items():
        for at_bat_idx, at_bat in at_bats.items():
            pitches = at_bat.pitches
            if len(pitches) == 0:
                continue
            at_bat_dict = asdict(at_bat)
            at_bat_level_info = pd.Series({'game_id': game_id} | at_bat_dict).drop('pitches')
            at_bat_df = pd.concat([at_bat_level_info]*len(pitches), axis=1).T
            at_bat_df['balls'] = [pitch.count['balls'] for pitch in pitches]
            at_bat_df['strikes'] = [pitch.count['strikes'] for pitch in pitches]
            at_bat_df['pitch_type'] = [pitch.pitch_type for pitch in pitches]        
            at_bat_df['at_bat'] = at_bat_idx
            at_bat_dfs.append(at_bat_df)
    season_df = pd.concat(at_bat_dfs)
    season_df = season_df.loc[(season_df['balls'] <= 3) & (season_df['strikes'] <= 2)] # drop a few entry errors
    season_df.reset_index(drop=True, inplace=True)
    return season_df

## Clean up dataframe
def clean_pitch_type(df):
    """Clean up/rename values in pitch type variable"""
    common_pitches = ['Four-Seam Fastball', 'Fastball', 'Slider', 'Sinker', 'Changeup', 'Curveball', 'Cutter', 'Knuckle Curve', 'Splitter']
    df.loc[~df['pitch_type'].isin(common_pitches), 'pitch_type'] = "Other"    
    df['pitch_type'].replace("Four-Seam Fastball", "Fastball", inplace=True) # merge 4-seam and regular fastball
    df['pitch_type'].replace("Knuckle Curve", "Knuckle_Curve", inplace=True)
    df['pitch_type'] = df['pitch_type'].str.lower()
    return df

def get_count(df):
    """Add column for count to dataframe"""
    df['count'] = ("(" + df['balls'].astype(str) + "," + df['strikes'].astype(str) + ")")
    return df 

def get_pitching_lead(df):
    """Return lead of pitching team"""
    home_score, away_score = df['home_score'], df['away_score']
    df['pitching_lead'] = np.where(df['top'], home_score-away_score, away_score-home_score)
    return df 

def get_last_pitch_type(df):
    """Add column for lagged pitch type"""
    df['last_pitch_type'] = df['pitch_type'].shift(1)
    df.loc[(df['count'] == "(0,0)"), 'last_pitch_type'] = None # if new at bat, strip last count
    return df

def clean_data_frame(season_df):
    """Clean up dataframe"""
    season_df = clean_pitch_type(season_df)
    season_df = get_count(season_df)
    season_df = get_last_pitch_type(season_df)
    return season_df 
