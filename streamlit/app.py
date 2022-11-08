import streamlit as st
import pandas as pd
import numpy as np 
import time 
import os

### Page Config
st.set_page_config(
    page_title="PitchCast",
    page_icon="⚾️"
)

### Page Content
@st.cache
def load_2022_data():
    df = pd.read_csv("data/2022.csv")
    return df
    
df = load_2022_data()
df_last = df.groupby('pitcher_name').last()
df_last = df_last.loc[df_last['nth_season_pitch'] >= 100] # restrict to pitchers with >= 100 pitches
pitchers = list(sorted(df_last.index.unique()))
pitchers_lefty = df_last['pitcher_lefty']
pitchers_rates = df_last[['fastball_rate', 'sinker_rate', 'slider_rate', 'changeup_rate', 'knuckle_curve_rate', 'curveball_rate', 'cutter_rate', 'splitter_rate', 'other_rate']]

## Title
st.title("⚾️ PitchCast")

## About
st.markdown("# About")
st.markdown("- TODO: Explain the project")
st.markdown("- TODO: Point to code")
st.markdown("- TODO: Explain that this is simplified model (no ab pitch type counts, lag ball/strike, etc.")
st.markdown("- TODO: Explain that this is based on 2022 rates")
st.markdown("- TODO: Make simplified model")

## Demo
st.markdown("# Demo")
st.write("Select the features below and PitchCast will estimate the probabilities of each pitch type")


st.markdown("## Pitcher/Batter")
pitcher_batter = st.columns(2)
st.markdown("## Count")
count = st.columns(2)
st.markdown("## Game State")
inning_outs = st.columns(3)
runners = st.columns(3)
scoreboard = st.columns(2)
st.markdown("## Pitch Counts")
pitch_counts = st.columns(3)
st.markdown("## Pitch Type Projection")

# pitcher/batter
pitcher = pitcher_batter[0].selectbox("Pitcher:", pitchers)
pitcher_lefty = pitchers_lefty[pitcher]
pitcher_rates = pitchers_rates.loc[pitcher]
batter_hand = pitcher_batter[1].radio("Batter Hand", ['Righty', 'Lefty'])

# Count
balls = count[0].slider("Balls:", 0, 3)
strikes = count[1].slider("Strikes:", 0, 2)

# Gamestate
inning = inning_outs[0].slider("Inning:", 1, 18)
top_bottom = inning_outs[1].radio("Top/Bottom:", ['Top', 'Bottom'])
outs = inning_outs[2].slider("Outs:", 0, 2)
runner1 = runners[0].checkbox("Runner on First")
runner2 = runners[1].checkbox("Runner on Second")
runner3 = runners[2].checkbox("Runner on Third")
home_score = scoreboard[0].slider("Home Score", 0, 20)
away_score = scoreboard[1].slider("Away Score", 0, 20)

# Pitch Counts
game_pitch_count = pitch_counts[0].slider("Total Pitch Count (by current pitcher):", 0, 125)
inning_pitch_count = pitch_counts[1].slider("Inning Pitch Count:", 0, 50)
at_bat_pitch_count = pitch_counts[2].slider("At Bat Pitch Count:", 0, 15)
