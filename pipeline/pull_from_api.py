import statsapi
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Pitch:
    pitch_type: str
    count: Tuple[Dict[str, int]]

@dataclass
class AtBat:
    inning: int
    top: bool
    home_score: int
    away_score: int
    outs: int
    pitcher_name: str
    pitcher_id: int
    pitcher_lefy: bool
    batter_name: str
    batter_id: int
    batter_leftie: bool
    pitches: List[Pitch]

## Season-Level
def get_season_data(season):
    """Get season-level data"""
    game_ids = get_game_ids(season)
    season_data = dict()
    for game_id in game_ids: # loop through games
        game_data = get_game_data(game_id)
        season_data[game_id] = game_data
    return season_data

def get_game_ids(season):
    """Helper: get game IDs for season"""
    start_date, end_date = get_season_dates(season)
    schedule = statsapi.schedule(start_date=start_date, end_date=end_date)
    schedule = [game for game in schedule if game['home_name'] not in (['American League All-Stars', 'National League All-Stars']) and game['status'] == "Final"]
    game_ids = list({game['game_id'] for game in schedule})
    return game_ids

def get_season_dates(season):
    """Helper: get season start and end dates"""
    season_dates = statsapi.get('season', {'seasonId':season, 'sportId':1})['seasons'][0]
    start_date = season_dates['regularSeasonStartDate']
    end_date = season_dates['regularSeasonEndDate']
    return start_date, end_date

## Game-Level
def get_game_data(game_id):
    """Get game-level data"""
    play_by_play = statsapi.get("game_playByPlay", params={'gamePk':game_id})['allPlays']
    game_data = dict()
    for at_bat in play_by_play:
        at_bat_data = get_at_bat_data(at_bat)
        at_bat_idx = at_bat['atBatIndex']
        game_data[at_bat_idx] = at_bat_data
    return game_data

## At Bat-Level
def get_at_bat_data(at_bat):
    """Get at bat-level data"""
    matchup = at_bat['matchup']
    inning = at_bat['about']['inning']
    top = at_bat['about']['isTopInning']
    home_score, away_score = get_score(at_bat) # score going into this at_bat
    outs = at_bat['playEvents'][0]['count']['outs'] # outs going into this at_bat
    pitcher_name = matchup['pitcher']['fullName']
    pitcher_id = matchup['pitcher']['id'] 
    pitcher_lefty = matchup['pitchHand']['code'] == "L"
    batter_name = matchup['batter']['fullName']
    batter_id = matchup['batter']['id']
    batter_lefty = matchup['batSide']['code'] == "L"

    pitches = get_pitch_data(at_bat)
    at_bat = AtBat(inning, top, home_score, away_score, outs, pitcher_name, pitcher_id, pitcher_lefty, batter_name, batter_id, batter_lefty, pitches)
    return at_bat

def get_score(at_bat):
    """Helper: get the score going into a at_bat"""
    home_score = at_bat['result']['homeScore'] # home score *after* this at_bat
    away_score = at_bat['result']['awayScore'] # away score *after* this at_bat
    at_bat_runs = sum([runner['movement']['end'] == "score" for runner in at_bat['runners']]) # runs scored on at_bat
    if at_bat_runs > 0:
        top_of_inning = at_bat['about']['isTopInning'] # top or bottom of inning
        if top_of_inning:
            away_score -= at_bat_runs
        else:
            home_score -= at_bat_runs
    return home_score, away_score

## Pitch-Level
def get_pitch_data(at_bat): 
    """Get pitch-level data"""
    pitches = list()
    count = {'balls': 0, 'strikes': 0}
    for event in at_bat['playEvents']:
        if not event['isPitch']:
            continue
        pitch_details = event['details']
        pitch_type = get_pitch_type(pitch_details)
        pitch = Pitch(pitch_type=pitch_type, count=count)
        pitches.append(pitch)
        count = event['count']
    return pitches

def get_pitch_type(pitch_details):
    """Helper: get pitch type"""
    if pitch_details['description'] == "Automatic Ball":
        pitch_type = "Automatic Ball"
    else:
        pitch_type = pitch_details.get('type')
        if pitch_type is not None: # pitches are occasionally missing a type
            pitch_type = pitch_type['description']
    return pitch_type

