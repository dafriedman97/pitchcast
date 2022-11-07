import pandas as pd
import numpy as np

def create_features(season_df):
    """Main function—clean up and augment dataframe"""
    season_df = clean_pitch_type(season_df)
    season_df = get_count(season_df)
    season_df = get_last_pitch_type(season_df)
    season_df = get_previous_balls_and_strikes(season_df)
    season_df = get_pitch_count(season_df)
    season_df = get_inning_pitch_count(season_df)
    season_df = get_at_bat_pitch_count(season_df)
    season_df = get_at_bat_pitches(season_df)
    return season_df 

def clean_pitch_type(df):
    """Clean up/rename values in pitch type variable"""
    df['raw_pitch_type'] = df['pitch_type']
    common_pitches = ['Four-Seam Fastball', 'Fastball', 'Slider', 'Sinker', 'Changeup', 'Curveball', 'Cutter', 'Knuckle Curve', 'Splitter']
    df.loc[~df['pitch_type'].isin(common_pitches), 'pitch_type'] = "Other"
    df['pitch_type'].replace("Four-Seam Fastball", "Fastball", inplace=True)
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
    df['pitch_type_lag_1'] = df['pitch_type'].shift(1)
    df['pitch_type_lag_2'] = df['pitch_type'].shift(2)
    df.loc[df['batter_id'] != df['batter_id'].shift(1), 'pitch_type_lag_1'] = "none" # no lag 1 pitch type for first pitch of at bat
    df.loc[df['batter_id'] != df['batter_id'].shift(2), 'pitch_type_lag_2'] = "none" # no lag 2 pitch type for first/second pitch of at bat
    return df

def get_previous_balls_and_strikes(df):
    """Add balls and strikes in previous pitches during the at bat"""
    for i in range(1, 4):
        df[f'lag_{i}_ball'] = df['balls'] > df['balls'].shift(i)
        df[f'lag_{i}_strike'] = df['strikes'] > df['strikes'].shift(i)
        df.loc[df['at_bat_index'].shift(i) != df['at_bat_index'], f'lag_{i}_ball'] = False # lag_i_ball is False if ith lagged pitch was different at bat
        df.loc[df['at_bat_index'].shift(i) != df['at_bat_index'], f'lag_{i}_strike'] = False # lag_i_strike is False if ith lagged pitch was different at bat
    return df

def get_pitch_count(df):
    """Add number of pitches pitcher has thrown this game"""
    df['pitch_count'] = df.groupby(['game_id', 'pitcher_id']).cumcount()
    return df

def get_inning_pitch_count(df):
    """Add number of pitches pitcher has thrown this inning"""
    df['inning_pitch_count'] = df.groupby(['game_id', 'inning', 'pitcher_id']).cumcount()
    return df

def get_at_bat_pitch_count(df):
    """Add number of pitches thrown already in this at bat"""
    df['ab_pitch_count'] = df.groupby(['game_id', 'at_bat_index']).cumcount()
    return df

def get_at_bat_pitches(df):
    """Add number of pitches of each type so far in at bat"""
    pitch_type_dummies = pd.get_dummies(df['pitch_type'], prefix="ab") # get pitch types as dummies
    pitch_type_dummies = pitch_type_dummies.join(df[['game_id', 'at_bat_index']]) # add in at bat index
    pitch_type_counts = pitch_type_dummies.groupby(['game_id', 'at_bat_index']).cumsum() # get running count of pitch types
    count_columns = [c + "_count" for c in pitch_type_counts.columns]
    pitch_type_counts.columns = count_columns
    pitch_type_counts[['game_id', 'at_bat_index']] = pitch_type_dummies[['game_id', 'at_bat_index']] # add at bat index to running counts
    pitch_type_counts[count_columns] = pitch_type_counts[count_columns].shift(1, fill_value=0) # shift counts to get counts prior to pitch
    pitch_type_counts.loc[pitch_type_counts['at_bat_index'] != pitch_type_counts['at_bat_index'].shift(1), count_columns] = 0
    df = df.join(pitch_type_counts[count_columns])
    return df
