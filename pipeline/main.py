import pull_from_api, convert_to_dataframe, feature_engineering, add_pitch_rates, pitcher_level_info
import os
import argparse

## Main
def main(season):
    """Collect, clean, and augment season data"""
    season_data = pull_from_api.get_season_data(season) # pull season data from api
    season_df = convert_to_dataframe.create_raw_data_frame(season_data) # convert into dataframe 
    season_df = feature_engineering.create_features(season_df) # add created features
    season_df = add_pitch_rates.add_pitcher_pitch_rates(season_df) # add pitcher's frequency of pitch-types
    return season_df 

if __name__ == "__main__":    
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--season", type=str)
    parser.add_argument("-o", "--overwrite", action='store_true')
    parser.add_argument("-p", "--pitcher_info", action='store_true')
    args = parser.parse_args()
    season = args.season
    pitcher_info = args.pitcher_info
    overwrite = args.overwrite

    # Get paths
    pipeline_dir = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.dirname(pipeline_dir), "data")
    season_path = os.path.join(data_dir, f"{season}.csv")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    # Get data
    if overwrite or not os.path.exists(season_path):
        season_df = main(season)
        season_df.to_csv(season_path, index=False)
        if pitcher_info:
            pitcher_info = pitcher_level_info.get_pitcher_level_info(season_df)
            pitcher_info.to_csv(os.path.join(data_dir, f"pitcher_level_info_{season}.csv"), index=False)
            