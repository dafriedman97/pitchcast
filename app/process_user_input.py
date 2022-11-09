pitch_types = ['fastball', 'curveball', 'sinker', 'cutter', 'changeup', 'slider', 'splitter', 'knuckle_curve', 'other']
counts = [
    'count_(0, 0)', 'count_(0, 1)', 'count_(0, 2)',
    'count_(1, 0)', 'count_(1, 1)', 'count_(1, 2)',
    'count_(2, 0)', 'count_(2, 1)', 'count_(2, 2)',
    'count_(3, 0)', 'count_(3, 1)', 'count_(3, 2)'
]

def get_features():
    """List features"""

    # Plain/direct features
    game_state = ['inning', 'top', 'outs']
    runners = ['runner_1', 'runner_2', 'runner_3']
    scores = ['home_score', 'away_score']
    pitch_counts = ['pitch_count', 'inning_pitch_count', 'ab_pitch_count']
    hands = ['pitcher_lefty', 'batter_lefty']
    features = game_state + runners + scores + pitch_counts + hands

    # Pitch type
    pitch_type_rates = [pitch_type + "_rate" for pitch_type in pitch_types]
    features.extend(pitch_type_rates)

    # Dummy columns
    features.extend(counts)

    # Return
    return features

def create_X(game_state, runners, scores, pitch_counts, lefties, pitcher_rates, balls, strikes):
    """Create single sample for prediction"""
    
    pitch_rates = list(pitcher_rates[[f"{pitch}_rate" for pitch in pitch_types]])
    count_dummies = [int(count == f"count_({balls}, {strikes})") for count in counts]
    features = game_state + runners + scores + pitch_counts + lefties + pitch_rates + count_dummies
    return features