import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import gradio as gr

# --- 1. Setup and Loading ---
print("✅ Loading all saved models, feature lists, and master data...")

try:
    # --- Define File Paths ---
    PROJECT_FOLDER = r'C:\Users\angkon\Desktop\Project'
    DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'Data')
    
    # --- Load DataFrames ---
    DF_PATH = os.path.join(PROJECT_FOLDER, 'final_model_dataset.pkl')
    df_master = pd.read_pickle(DF_PATH)
    COMPETITIONS_PATH = os.path.join(DATA_FOLDER, 'competitions.csv')
    df_competitions = pd.read_csv(COMPETITIONS_PATH)

    # --- Load Models ---
    print("   - Loading trained models...")
    outcome_model = xgb.XGBClassifier()
    outcome_model.load_model(os.path.join(PROJECT_FOLDER, 'final_balanced_xgboost_model.json'))
    
    # NOTE: The over_under_2_5_model has been removed as it's no longer needed.
    # The prediction will be derived directly from the xG models.
    
    btts_model = xgb.XGBClassifier()
    btts_model.load_model(os.path.join(PROJECT_FOLDER, 'btts_model.json'))
    
    # --- Define Feature Lists ---
    print("   - Defining feature lists for models...")
    all_numeric_cols = df_master.select_dtypes(include=np.number)
    features_to_drop_outcome = [
        'game_id', 'home_club_id', 'away_club_id', 'home_coach_id', 'away_coach_id',
        'home_club_goals', 'away_club_goals', 'total_goals', 'result_encoded',
        'goals_home', 'assists_home', 'yellow_cards_home', 'red_cards_home',
        'goals_away', 'assists_away', 'yellow_cards_away', 'red_cards_away'
    ]
    model_features = [col for col in all_numeric_cols.columns if col not in features_to_drop_outcome]

    secondary_features = [
        'elo_diff', 'coach_elo_diff', 'market_value_diff', 'market_value_ratio',
        'prob_H', 'prob_D', 'prob_A', 'starting_xi_value_diff',
        'formation_attack_diff', 'clv_home', 'clv_draw', 'clv_away',
        'poisson_xg_diff', 'home_cs_form', 'home_fts_form',
        'away_cs_form', 'away_fts_form'
    ]
    secondary_features = [col for col in secondary_features if col in df_master.columns]

    # --- Retrain xG models for this session ---
    print("   - Training temporary xG models...")
    X_xg = df_master[secondary_features].fillna(0)
    y_home_goals = df_master['home_club_goals']
    y_away_goals = df_master['away_club_goals']
    
    xg_home_model = xgb.XGBRegressor(objective='count:poisson', random_state=42).fit(X_xg, y_home_goals)
    xg_away_model = xgb.XGBRegressor(objective='count:poisson', random_state=42).fit(X_xg, y_away_goals)
        
    print("   - All models and data loaded successfully.")

except Exception as e:
    raise RuntimeError(f"A required model or file was not found. Please check your file paths. Error: {e}")


# --- 2. Prepare Data for the User Interface ---
league_name_map = df_competitions.set_index('competition_id')['name'].to_dict()
teams_by_league = {}
for comp_id in df_master['competition_id'].unique():
    league_name = league_name_map.get(comp_id, comp_id)
    league_teams = pd.concat([
        df_master[df_master['competition_id'] == comp_id]['home_team'],
        df_master[df_master['competition_id'] == comp_id]['away_team']
    ]).dropna().unique()
    league_teams.sort()
    teams_by_league[league_name] = list(league_teams)
print("✅ Data for interactive dropdowns is ready.")


# --- 3. The Prediction Function ---
def predict_match(home_team, away_team):
    """Predicts the outcome of a football match using our saved models."""
    if not home_team or not away_team:
        return "Please select both a home and away team."
    if home_team == away_team:
        return "ERROR: Home and Away teams cannot be the same."

    home_stats = df_master[(df_master['home_team'] == home_team) | (df_master['away_team'] == home_team)].sort_values(by='date').iloc[-1]
    away_stats = df_master[(df_master['away_team'] == away_team) | (df_master['home_team'] == away_team)].sort_values(by='date').iloc[-1]

    feature_vector = pd.Series(index=model_features, dtype='float64')

    feature_vector['elo_home'] = home_stats['elo_home'] if home_stats['home_team'] == home_team else home_stats['elo_away']
    feature_vector['elo_away'] = away_stats['elo_away'] if away_stats['away_team'] == away_team else away_stats['elo_home']
    feature_vector['coach_elo_home'] = home_stats['coach_elo_home'] if home_stats['home_team'] == home_team else home_stats['coach_elo_away']
    feature_vector['coach_elo_away'] = away_stats['coach_elo_away'] if away_stats['away_team'] == away_team else away_stats['elo_home']
    feature_vector['elo_diff'] = feature_vector['elo_home'] - feature_vector['elo_away']
    feature_vector['coach_elo_diff'] = feature_vector['coach_elo_home'] - feature_vector['coach_elo_away']
    
    for feat in feature_vector.index:
        if feat in home_stats:
            feature_vector[feat] = home_stats[feat]
    feature_vector.fillna(0, inplace=True)
    feature_df = feature_vector.to_frame().T

    X_pred_outcome = feature_df[model_features]
    X_pred_secondary = feature_df[secondary_features]

    outcome_pred_encoded = outcome_model.predict(X_pred_outcome)[0]
    outcome_map = {2: "Home Win", 1: "Draw", 0: "Away Win"}
    match_winner = outcome_map[outcome_pred_encoded]

    xg_home = xg_home_model.predict(X_pred_secondary)[0]
    xg_away = xg_away_model.predict(X_pred_secondary)[0]
    total_xg = xg_home + xg_away

    # --- FIX: The Over/Under prediction is now derived directly from the total xG ---
    over_under = "Over 2.5 Goals" if total_xg > 2.5 else "Under 2.5 Goals"

    btts_pred = btts_model.predict(X_pred_secondary)[0]
    btts = "Yes" if btts_pred == 1 else "No"
    
    predictions = {
        "Match Result": match_winner,
        "Total Expected Match Goals": f"{total_xg:.2f}",
        "Over/Under 2.5": over_under,
        "Both Teams to Score": btts
    }
    
    # Format the output nicely
    output_text = f"Predictions for: {home_team} vs. {away_team}\n" + "-"*40
    for key, value in predictions.items():
        output_text += f"\n- {key}: {value}"
    return output_text

# --- 4. Gradio Interface Logic ---
def update_team_dropdowns(league):
    """Dynamically updates team choices based on selected league."""
    teams = teams_by_league.get(league, [])
    return gr.update(choices=teams, value=teams[0] if teams else None), gr.update(choices=teams, value=teams[1] if len(teams) > 1 else None)

# --- 5. Build the Gradio Application ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚽ Football Match Predictor")
    gr.Markdown("Select a league, then choose a home and away team to generate predictions.")
    
    with gr.Row():
        league_dd = gr.Dropdown(choices=sorted(teams_by_league.keys()), label="Select League", value=sorted(teams_by_league.keys())[0])
    with gr.Row():
        home_dd = gr.Dropdown(label="Home Team")
        away_dd = gr.Dropdown(label="Away Team")
        
    predict_btn = gr.Button("Generate Prediction", variant="primary")
    output_txt = gr.Textbox(label="Model Predictions", lines=5)

    # Link the components
    league_dd.change(fn=update_team_dropdowns, inputs=league_dd, outputs=[home_dd, away_dd])
    predict_btn.click(fn=predict_match, inputs=[home_dd, away_dd], outputs=output_txt)
    
    # Pre-populate the team dropdowns on launch
    demo.load(fn=update_team_dropdowns, inputs=league_dd, outputs=[home_dd, away_dd])

print("🚀 Launching the Gradio application...")
demo.launch()

