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
pitchers_lefty = pitcher_data['pitcher_lefty']
pitchers_rates = pitcher_data[
    ['fastball_rate', 'sinker_rate', 'slider_rate', 'changeup_rate', 'knuckle_curve_rate', 'curveball_rate', 'cutter_rate', 'splitter_rate', 'other_rate']
]
pitchers = sorted(list(pitchers_rates.index))

model = load_model()

### Page Content ###

## Title
st.title("⚾️ PitchCast")

## About
st.markdown("# About")
st.markdown("- TODO: Explain the project")
st.markdown("- TODO: Point to code")
st.markdown("- TODO: Explain that this is simplified model (no ab pitch type counts, lag ball/strike, etc.")
st.markdown("- TODO: Explain that this is based on 2022 rates")

## Demo
st.markdown("# Demo") 
st.write("Select the features below and PitchCast will estimate the probabilities of each pitch type")

input, output = st.columns(2)

with input:
    st.markdown("## Features")
    with st.expander("Pitcher/Batter"):
        pitcher = st.selectbox("Pitcher:", pitchers)
        pitcher_lefty = pitchers_lefty[pitcher]
        pitcher_rates = pitchers_rates.loc[pitcher]
        batter_hand = st.radio("Batter Hand", ['Righty', 'Lefty'])
        lefties = [pitcher_lefty == "Lefty", batter_hand == "Lefty"]        
    with st.expander("Count"):
        balls = st.slider("Balls:", 0, 3)
        strikes = st.slider("Strikes:", 0, 2)
    with st.expander("Game State"):
        home_score = st.slider("Home Score", 0, 20)
        away_score = st.slider("Away Score", 0, 20)
        inning = st.slider("Inning:", 1, 18)
        top_bottom = st.radio("Top/Bottom:", ['Top', 'Bottom'])
        outs = st.slider("Outs:", 0, 2)
        runner_1 = st.checkbox("Runner on First")
        runner_2 = st.checkbox("Runner on Second")
        runner_3 = st.checkbox("Runner on Third")
        game_state = [inning, top_bottom == "Top", outs]
        runners = [runner_1, runner_2, runner_3]
        scores = [home_score, away_score]
    with st.expander("Pitch Counts"):
        game_pitch_count = st.slider("Total Pitch Count (by current pitcher):", 0, 125)
        inning_pitch_count = st.slider("Inning Pitch Count:", 0, 50)
        at_bat_pitch_count = st.slider("At Bat Pitch Count:", 0, 15)
        pitch_counts = [game_pitch_count, inning_pitch_count, at_bat_pitch_count]        

with output:
    st.markdown("## Pitch Type Probabilities")
    # Create prediction 
    X = ui.create_X(game_state, runners, scores, pitch_counts, lefties, pitcher_rates, balls, strikes)
    X = np.array(X).reshape(1, -1)
    pitch_type_probs = model.predict(X)[0]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(y=ui.pitch_types, x=pitch_type_probs, ax=ax, palette='Set2')
    ax.set_yticklabels([t.get_text().replace("_rate", "") for t in ax.get_yticklabels()], rotation=20)
    ax.set_xlabel("Probability")
    ax.grid()
    sns.despine()
    st.pyplot(fig)
