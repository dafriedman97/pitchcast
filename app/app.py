import streamlit as st
import pandas as pd
import numpy as np 
from tensorflow import keras
import process_user_input as ui
import seaborn as sns
import matplotlib.pyplot as plt

### Page Config ###
st.set_page_config(
    page_title="PitchCast",
    page_icon="⚾️",
    layout="wide"
)

### Load Model + Data ###
@st.cache
def load_pitcher_data():
    df = pd.read_csv("data/pitcher_level_info_2022.csv").set_index('pitcher_name')
    return df

@st.cache(allow_output_mutation=True)
def load_model():
    model = keras.models.load_model("model/saved_models/fit_model_simple.h5")
    return model
    
pitcher_data = load_pitcher_data()
pitch_types = ['fastball', 'curveball', 'sinker', 'cutter', 'changeup', 'slider', 'splitter', 'knuckle_curve', 'other']
pitchers_lefty = pitcher_data['pitcher_lefty']
pitchers_rates = pitcher_data[[f"{pitch}_rate" for pitch in pitch_types]]
pitchers = sorted(list(pitchers_rates.index))

model = load_model()

### Page Content ###

## Title
st.title("⚾️ PitchCast")

## About
st.markdown("# About")
st.markdown("""This app allows users to interactively engage with [PitchCast](https://github.com/dafriedman97/pitchcast), a machine learning model designed to predict the next pitch type in MLB baseball games.""")
st.markdown("""To visualize pitch type predictions, input the features below, such as the pitcher and the game-state, and observe the probabilities of each pitch type.""")
st.markdown("""Note that this is app uses a simplified version of the PitchCast model which does not require users to e.g. input the pitch types of previous pitches in the current at-bat.""")

## Demo
st.markdown("# Demo") 
st.write("Select the features below and PitchCast will estimate the probabilities of each pitch type. (Predictions are available for pitchers with 100 or more pitches in the 2022 regular season.)")

input, output = st.columns(2)

with input:
    st.markdown("## Features")
    with st.expander("Pitcher/Batter"):
        pitcher = st.selectbox("Pitcher:", pitchers)
        pitcher_lefty = pitchers_lefty[pitcher]
        pitcher_rates = pitchers_rates.loc[pitcher]
        batter_hand = st.radio("Batter Hand", ['Righty', 'Lefty'])
        lefties = [pitcher_lefty == "Lefty", batter_hand == "Lefty"]        
    with st.expander("Game State"):
        home_score = st.number_input("Home Score", 0, 20)
        away_score = st.number_input("Away Score", 0, 20)
        inning = st.number_input("Inning:", 1, 18)
        top_bottom = st.radio("Top/Bottom:", ['Top', 'Bottom'])
        outs = st.number_input("Outs:", 0, 2)
        runner_1 = st.checkbox("Runner on First")
        runner_2 = st.checkbox("Runner on Second")
        runner_3 = st.checkbox("Runner on Third")
        game_state = [inning, top_bottom == "Top", outs]
        runners = [runner_1, runner_2, runner_3]
        scores = [home_score, away_score]
    with st.expander("Count"):
        balls = st.number_input("Balls:", min_value=0, max_value=3)
        strikes = st.number_input("Strikes:", min_value=0, max_value=2)
    with st.expander("Pitch Counts"):
        st.write("Note: pitch counts refer to pitches thrown by the _current_ pitcher")
        min_pitches = int(balls+strikes)
        at_bat_pitch_count = st.number_input("At Bat Pitch Count:", min_value=min_pitches, max_value=15)
        min_inning_pitches = int(at_bat_pitch_count)
        inning_pitch_count = st.number_input("Inning Pitch Count:", min_value=min_inning_pitches, max_value=50)
        min_game_pitches = int(inning_pitch_count)
        game_pitch_count = st.number_input("Total Pitch Count:", min_value=min_game_pitches, max_value=125)
        pitch_counts = [game_pitch_count, inning_pitch_count, at_bat_pitch_count]        

with output:
    st.markdown("## Pitch Type Probabilities")
    # Create prediction 
    X = ui.create_X(game_state, runners, scores, pitch_counts, lefties, pitcher_rates, balls, strikes)
    X = np.array(X).reshape(1, -1)
    pitch_type_probs = model.predict(X)[0]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = sns.color_palette("flare", 9)
    sns.barplot(y=ui.pitch_types, x=pitch_type_probs, ax=ax, palette=palette)
    ax.set_yticklabels([t.get_text().replace("_rate", "") for t in ax.get_yticklabels()], rotation=20)
    ax.set_xlabel("Probability")
    ax.grid()
    sns.despine()
    st.pyplot(fig)
