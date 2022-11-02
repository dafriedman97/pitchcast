import pandas as pd
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
