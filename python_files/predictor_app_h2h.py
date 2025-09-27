import gradio as gr
import xgboost as xgb
import pandas as pd
import os
import numpy as np

print("\n✅ Building the final Gradio application with H2H features...")

try:
    # --- 1. Load All Necessary Data and Models ---
    print("   - Loading all required data and models...")
    PROJECT_FOLDER = r'C:\Users\angkon\Desktop\Project'
    DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'Data')

    # Load the new dataset that INCLUDES H2H features
    DF_PATH = os.path.join(PROJECT_FOLDER, 'final_model_dataset_h2h.pkl')
    df = pd.read_pickle(DF_PATH)
    df.sort_values(by='date', inplace=True)
    
    COMPETITIONS_PATH = os.path.join(DATA_FOLDER, 'competitions.csv')
    df_competitions = pd.read_csv(COMPETITIONS_PATH)

    # Load the H2H models
    champion_model_h2h = xgb.XGBClassifier()
    champion_model_h2h.load_model(os.path.join(PROJECT_FOLDER, 'champion_model_h2h.json'))
    
    # --- FIX: Removed the separate over_under_2_5_model_h2h ---
    
    btts_model_h2h = xgb.XGBClassifier()
    btts_model_h2h.load_model(os.path.join(PROJECT_FOLDER, 'btts_model_h2h.json'))

    # --- 2. Recreate Feature Lists and Split Data ---
    print("   - Recreating feature lists...")
    TARGET = 'result_encoded'
    X = df.select_dtypes(include=np.number).drop(columns=[
        'game_id', 'home_club_id', 'away_club_id', 'home_coach_id', 'away_coach_id',
        'home_club_goals', 'away_club_goals', 'total_goals', 'result_encoded',
        'goals_home', 'assists_home', 'yellow_cards_home', 'red_cards_home',
        'goals_away', 'assists_away', 'yellow_cards_away', 'red_cards_away'
    ]).fillna(0)
    y = df[TARGET]
    split_index = int(len(df) * 0.80)
    X_train, _ = X.iloc[:split_index], X.iloc[split_index:]

    # --- 3. Prepare UI Data and Retrain xG Models ---
    league_name_map = df_competitions.set_index('competition_id')['name'].to_dict()
    teams_by_league = {}
    for comp_id in df['competition_id'].unique():
        league_name = league_name_map.get(comp_id, comp_id)
        league_teams = pd.concat([
            df[df['competition_id'] == comp_id]['home_team'],
            df[df['competition_id'] == comp_id]['away_team']
        ]).dropna().unique()
        league_teams.sort()
        teams_by_league[league_name] = list(league_teams)
    print("   - Data for interactive dropdowns is ready.")

    # Retrain the temporary xG models for this session
    X_xg = df[X_train.columns].fillna(0) 
    y_home_goals = df['home_club_goals']
    y_away_goals = df['away_club_goals']
    xg_home_model_h2h = xgb.XGBRegressor(objective='count:poisson', random_state=42).fit(X_xg, y_home_goals)
    xg_away_model_h2h = xgb.XGBRegressor(objective='count:poisson', random_state=42).fit(X_xg, y_away_goals)
    print("   - All components loaded successfully.")

except Exception as e:
    print(f"❌ An error occurred during data preparation: {e}")


# --- 4. The Prediction Function ---
def predict_match_h2h(home_team, away_team):
    """Predicts match outcomes using the new H2H models."""
    if not home_team or not away_team: return "Please select both teams."
    if home_team == away_team: return "ERROR: Teams cannot be the same."

    home_stats_proxy = df[(df['home_team'] == home_team) | (df['away_team'] == home_team)].sort_values(by='date').iloc[-1]
    h2h_stats_proxy = df[((df['home_team'] == home_team) & (df['away_team'] == away_team))].sort_values(by='date')
    if not h2h_stats_proxy.empty:
        h2h_stats_proxy = h2h_stats_proxy.iloc[-1]
    else:
        h2h_stats_proxy = home_stats_proxy

    feature_vector = pd.Series(index=X_train.columns, dtype='float64')
    for feat in feature_vector.index:
        if feat in h2h_stats_proxy and pd.notna(h2h_stats_proxy[feat]):
            feature_vector[feat] = h2h_stats_proxy[feat]
        elif feat in home_stats_proxy:
            feature_vector[feat] = home_stats_proxy[feat]
    feature_df = feature_vector.fillna(0).to_frame().T[X_train.columns]

    outcome_pred = champion_model_h2h.predict(feature_df)[0]
    match_winner = {2: "Home Win", 1: "Draw", 0: "Away Win"}[outcome_pred]
    
    xg_home = xg_home_model_h2h.predict(feature_df)[0]
    xg_away = xg_away_model_h2h.predict(feature_df)[0]
    total_xg = xg_home + xg_away
    
    # --- FIX: Derive Over/Under prediction directly from total_xg for consistency ---
    over_under = "Over 2.5 Goals" if total_xg > 2.5 else "Under 2.5 Goals"
    
    btts_pred = btts_model_h2h.predict(feature_df)[0]
    btts = "Yes" if btts_pred == 1 else "No"
    
    predictions = {"Match Result": match_winner, "Total Expected Goals": f"{total_xg:.2f}",
                   "Over/Under 2.5": over_under, "Both Teams to Score": btts}
    output_text = f"Predictions for: {home_team} vs. {away_team}\n" + "-"*40
    for key, value in predictions.items():
        output_text += f"\n- {key}: {value}"
    return output_text

# --- 5. The Gradio UI ---
def update_team_dropdowns(league):
    teams = teams_by_league.get(league, [])
    return gr.update(choices=teams, value=teams[0] if teams else None), gr.update(choices=teams, value=teams[1] if len(teams) > 1 else None)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚽ Football Match Predictor By Angkon Biswas")
    gr.Markdown("Select a league, then choose a home and away team to generate predictions.")
    
    with gr.Row():
        league_dd = gr.Dropdown(choices=sorted(teams_by_league.keys()), label="Select League", value=sorted(teams_by_league.keys())[0])
    with gr.Row():
        home_dd = gr.Dropdown(label="Home Team")
        away_dd = gr.Dropdown(label="Away Team")
        
    predict_btn = gr.Button("Generate Prediction", variant="primary")
    output_txt = gr.Textbox(label="Model Predictions", lines=5)

    league_dd.change(fn=update_team_dropdowns, inputs=league_dd, outputs=[home_dd, away_dd])
    predict_btn.click(fn=predict_match_h2h, inputs=[home_dd, away_dd], outputs=output_txt)
    
    demo.load(fn=update_team_dropdowns, inputs=league_dd, outputs=[home_dd, away_dd])

print("🚀 Launching the Gradio application with H2H models...")
demo.launch()

