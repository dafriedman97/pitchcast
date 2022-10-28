import pull_from_api, convert_to_dataframe, add_pitch_rates
from add_pitch_rates import add_pitcher_pitch_rates
import os
import argparse

## Main
def main(season):
    """Collect, clean, and augment season data"""
    season_data = pull_from_api.get_season_data(season) # pull season data from api
    season_df = convert_to_dataframe.create_raw_data_frame(season_data) # convert into dataframe 
    season_df = convert_to_dataframe.clean_data_frame(season_df) # clean up    
    season_df = add_pitch_rates.add_pitcher_pitch_rates(season_df) # add pitcher's frequency of pitch-types
    return season_df 

if __name__ == "__main__":    
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--season", type=str)
    parser.add_argument("-o", "--overwrite", action='store_true')
    args = parser.parse_args()
    season = args.season
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
