import joblib
import json
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import ast
import os

# --- Define the absolute base directory of the script ---
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
app = FastAPI()

# --- Load All Necessary Files on Startup ---
model, teams_data, model_columns, team_name_map = None, {}, [], {}
print("\n--- API Server Starting Up ---")
try:
    print("Attempting to load all necessary files...")
    # Define absolute paths for all files
    model_path = os.path.join(BASE_DIR, 'model.pkl')
    teams_path = os.path.join(BASE_DIR, 'teams.json')
    columns_path = os.path.join(BASE_DIR, 'model_columns.json')
    map_path = os.path.join(BASE_DIR, 'Team_Map_Manual.txt')

    model = joblib.load(model_path)
    with open(teams_path, 'r') as f:
        teams_data = json.load(f)
    with open(columns_path, 'r') as f:
        model_columns = json.load(f)
    with open(map_path, 'r') as f:
        team_map_str = f.read().replace('team_name_map = ', '')
        team_name_map = ast.literal_eval(team_map_str)
        
    print(f"✅ All files loaded successfully. Found {len(teams_data)} leagues.")
except Exception as e:
    print(f"❌ CRITICAL Error during startup: A required file was not found or is corrupted. Please regenerate files from the notebook. Error: {e}")

# --- API request model ---
class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    odds_home: float
    odds_draw: float
    odds_away: float

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Football Match Result Predictor API"}

@app.get("/leagues")
async def get_leagues():
    return {"leagues": teams_data}

@app.post("/predict")
async def predict_match(request: PredictionRequest):
    if not all([model, model_columns, team_name_map]):
        return {"error": "Model or supporting files not loaded. Please check server logs."}
    
    # --- CORRECTED PREDICTION LOGIC ---
    try:
        # 1. Create a DataFrame with the exact columns the model was trained on
        input_df = pd.DataFrame(columns=model_columns)
        input_df.loc[0, :] = 0 # Initialize the row with a default value (e.g., 0)

        # 2. Translate clean names from the app back to the raw names the model expects
        home_team_raw = team_name_map.get(request.home_team, request.home_team)
        away_team_raw = team_name_map.get(request.away_team, request.away_team)
        
        # 3. Fill in the known values from the user request
        input_df.loc[0, 'home_team_name'] = home_team_raw
        input_df.loc[0, 'away_team_name'] = away_team_raw
        input_df.loc[0, 'odds_home'] = request.odds_home
        input_df.loc[0, 'odds_draw'] = request.odds_draw
        input_df.loc[0, 'odds_away'] = request.odds_away

        # 4. Fill in dummy values for other important features (the model's pipeline will handle the rest)
        dummy_features = {
            'h2h_home_win_percentage': 0.4, 
            'h2h_away_win_percentage': 0.3, 
            'h2h_draw_percentage': 0.3,
            'home_form_avg_goals_scored_5': 1.5, 
            'away_form_avg_goals_scored_5': 1.2
        }
        for col, value in dummy_features.items():
            if col in input_df.columns:
                input_df.loc[0, col] = value
                
        # 5. Ensure correct data types for categorical columns
        for col in input_df.select_dtypes(include=['object']).columns:
            input_df[col] = input_df[col].astype(str)

        # 6. Make the prediction
        prediction_code = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        prediction_text = result_map.get(prediction_code, "Unknown")
        prob_dict = {'away_win': probabilities[0], 'draw': probabilities[1], 'home_win': probabilities[2]}

        return {"prediction": prediction_text, "probabilities": prob_dict}
        
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return {"error": f"An error occurred during prediction: {e}"}

