import pandas as pd
import numpy as np
import xgboost as xgb
import os
import gradio as gr

# --- Constants and File Paths ---
PROJECT_FOLDER = r'C:\Users\angkon\Desktop\Project'
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'Data')
DF_PATH = os.path.join(PROJECT_FOLDER, 'final_model_dataset_h2h.pkl')
COMPETITIONS_PATH = os.path.join(DATA_FOLDER, 'competitions.csv')

# --- Definitive Feature Lists ---
FUNDAMENTAL_FEATURES = [
    'elo_diff', 'coach_elo_diff', 'market_value_diff', 'market_value_ratio',
    'starting_xi_value_diff', 'formation_attack_diff', 'home_cs_form', 'home_fts_form',
    'away_cs_form', 'away_fts_form', 'h2h_home_win_pct', 'h2h_home_draw_pct',
    'h2h_home_avg_gs', 'h2h_home_avg_gc'
]
FULL_FEATURES = FUNDAMENTAL_FEATURES + ['prob_H', 'prob_D', 'prob_A', 'clv_home', 'clv_draw', 'clv_away']

def setup_and_train_models():
    """
    Loads data, engineers features, and trains all necessary models in memory.
    """
    print("--- Initializing Application: Loading data and engineering features... ---")
    df = pd.read_pickle(DF_PATH)
    df.sort_values(by='date', inplace=True)
    
    # --- Engineer Features ---
    print("    - Engineering form, H2H, and ratio/difference features...")
    home_games_form = df[['game_id', 'date', 'home_club_id', 'home_club_goals', 'away_club_goals']].rename(
        columns={'home_club_id': 'team_id', 'home_club_goals': 'gs', 'away_club_goals': 'gc'})
    away_games_form = df[['game_id', 'date', 'away_club_id', 'away_club_goals', 'home_club_goals']].rename(
        columns={'away_club_id': 'team_id', 'away_club_goals': 'gs', 'home_club_goals': 'gc'})
    team_form_long = pd.concat([home_games_form, away_games_form]).sort_values(by='date')
    team_form_long['clean_sheet'] = (team_form_long['gc'] == 0).astype(int)
    team_form_long['failed_to_score'] = (team_form_long['gs'] == 0).astype(int)
    team_form_long['cs_form_10'] = team_form_long.groupby('team_id')['clean_sheet'].transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).mean())
    team_form_long['fts_form_10'] = team_form_long.groupby('team_id')['failed_to_score'].transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).mean())
    df = pd.merge(df, team_form_long[['game_id', 'team_id', 'cs_form_10', 'fts_form_10']], left_on=['game_id', 'home_club_id'], right_on=['game_id', 'team_id'], how='left').rename(columns={'cs_form_10': 'home_cs_form', 'fts_form_10': 'home_fts_form'})
    df = pd.merge(df, team_form_long[['game_id', 'team_id', 'cs_form_10', 'fts_form_10']], left_on=['game_id', 'away_club_id'], right_on=['game_id', 'team_id'], how='left').rename(columns={'cs_form_10': 'away_cs_form', 'fts_form_10': 'away_fts_form'})
    df.drop(columns=['team_id_x', 'team_id_y'], inplace=True)
    df['market_value_diff'] = df['total_market_value_home'] - df['total_market_value_away']
    df['market_value_ratio'] = df['total_market_value_home'] / (df['total_market_value_away'] + 1)

    # --- Train All Models ---
    print("--- Training all required models for this session... ---")
    y_outcome = df['result_encoded']
    y_ou = (df['total_goals'] > 2.5).astype(int)
    y_btts = ((df['home_club_goals'] > 0) & (df['away_club_goals'] > 0)).astype(int)
    
    split_index = int(len(df) * 0.80)
    y_train = y_outcome.iloc[:split_index]
    y_ou_train = y_ou.iloc[:split_index]
    y_btts_train = y_btts.iloc[:split_index]
    sample_weights = np.ones(len(y_train)); sample_weights[y_train == 1] *= 1.8

    X_fundamental = df[[col for col in FUNDAMENTAL_FEATURES if col in df.columns]].fillna(0)
    X_train_fund = X_fundamental.iloc[:split_index]
    fundamental_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42).fit(X_train_fund, y_train, sample_weight=sample_weights)
    print("    - Match Winner (Fundamentals) model trained.")

    X_full = df[[col for col in FULL_FEATURES if col in df.columns]].fillna(0)
    X_train_full = X_full.iloc[:split_index]
    full_feature_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42).fit(X_train_full, y_train, sample_weight=sample_weights)
    print("    - Match Winner (Full Features) model trained.")

    ou_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42).fit(X_train_fund, y_ou_train)
    print("    - Over/Under 2.5 model trained.")
    btts_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42).fit(X_train_fund, y_btts_train)
    print("    - BTTS model trained.")
    
    df_competitions = pd.read_csv(COMPETITIONS_PATH)
    league_name_map = df_competitions.set_index('competition_id')['name'].to_dict()
    teams_by_league = {league_name_map.get(comp_id, comp_id): sorted(pd.concat([df[df['competition_id'] == comp_id]['home_team'], df[df['competition_id'] == comp_id]['away_team']]).dropna().unique()) for comp_id in df['competition_id'].unique()}
    
    # --- Add Other/International and create a full team list ---
    teams_by_league['Other/International'] = []
    all_teams = sorted(pd.concat([df['home_team'], df['away_team']]).dropna().unique())
    
    return df, fundamental_model, full_feature_model, ou_model, btts_model, teams_by_league, all_teams

# --- Run Setup and Training on Start ---
df_master, fundamental_model, full_feature_model, ou_model, btts_model, teams_by_league, all_teams = setup_and_train_models()
print("✅ Application setup complete. UI is ready.")

# --- The Hybrid Prediction Function ---
def predict_match_hybrid(home_team, away_team, odds_h, odds_d, odds_a):
    if not home_team or not away_team: return "Please select both teams."
    if home_team == away_team: return "ERROR: Teams cannot be the same."

    # Verify that teams exist in the master dataframe
    if home_team not in all_teams or away_team not in all_teams:
        return "ERROR: One or both teams not found in the dataset. Please check spelling."

    prediction_row = pd.DataFrame(columns=FULL_FEATURES)
    home_stats = df_master[(df_master['home_team'] == home_team) | (df_master['away_team'] == home_team)].sort_values(by='date').iloc[-1]
    away_stats = df_master[(df_master['home_team'] == away_team) | (df_master['away_team'] == away_team)].sort_values(by='date').iloc[-1]
    
    home_elo = home_stats['elo_home'] if home_stats['home_team'] == home_team else home_stats['elo_away']
    away_elo = away_stats['elo_away'] if away_stats['away_team'] == away_team else away_stats['elo_home']
    home_coach_elo = home_stats['coach_elo_home'] if home_stats['home_team'] == home_team else home_stats['coach_elo_away']
    away_coach_elo = away_stats['coach_elo_away'] if away_stats['away_team'] == away_team else away_stats['coach_elo_home']
    home_market_value = home_stats['total_market_value_home'] if home_stats['home_team'] == home_team else home_stats['total_market_value_away']
    away_market_value = away_stats['total_market_value_away'] if away_stats['away_team'] == away_team else away_stats['total_market_value_home']
    home_xi_value = home_stats['starting_xi_value_home'] if home_stats['home_team'] == home_team else home_stats['starting_xi_value_away']
    away_xi_value = away_stats['starting_xi_value_away'] if away_stats['away_team'] == away_team else away_stats['starting_xi_value_home']
    prediction_row.loc[0, 'elo_diff'] = home_elo - away_elo
    prediction_row.loc[0, 'coach_elo_diff'] = home_coach_elo - away_coach_elo
    prediction_row.loc[0, 'market_value_diff'] = home_market_value - away_market_value
    prediction_row.loc[0, 'market_value_ratio'] = home_market_value / (away_market_value + 1)
    prediction_row.loc[0, 'starting_xi_value_diff'] = home_xi_value - away_xi_value
    prediction_row.loc[0, 'formation_attack_diff'] = (home_stats['formation_attack_score_home'] if home_stats['home_team'] == home_team else home_stats['formation_attack_score_away']) - \
                                                     (away_stats['formation_attack_score_away'] if away_stats['away_team'] == away_team else away_stats['formation_attack_score_home'])
    prediction_row.loc[0, 'home_cs_form'] = home_stats['home_cs_form'] if home_stats['home_team'] == home_team else home_stats['away_cs_form']
    prediction_row.loc[0, 'home_fts_form'] = home_stats['home_fts_form'] if home_stats['home_team'] == home_team else home_stats['away_fts_form']
    prediction_row.loc[0, 'away_cs_form'] = away_stats['away_cs_form'] if away_stats['away_team'] == away_team else away_stats['home_cs_form']
    prediction_row.loc[0, 'away_fts_form'] = away_stats['away_fts_form'] if away_stats['away_team'] == away_team else away_stats['home_fts_form']
    h2h_stats = df_master[((df_master['home_team'] == home_team) & (df_master['away_team'] == away_team))].sort_values(by='date')
    if not h2h_stats.empty:
        latest_h2h = h2h_stats.iloc[-1]
        for col in ['h2h_home_win_pct', 'h2h_home_draw_pct', 'h2h_home_avg_gs', 'h2h_home_avg_gc']:
            prediction_row.loc[0, col] = latest_h2h[col]

    if odds_h and odds_d and odds_a:
        print("    - Using Full-Feature Model with live odds...")
        prediction_row.loc[0, 'prob_H'] = 1 / odds_h
        prediction_row.loc[0, 'prob_D'] = 1 / odds_d
        prediction_row.loc[0, 'prob_A'] = 1 / odds_a
        prediction_row.loc[0, ['clv_home', 'clv_draw', 'clv_away']] = 0
        feature_df = prediction_row[FULL_FEATURES].fillna(0)
        model_to_use = full_feature_model
    else:
        print("    - Using Fundamentals-Only Model...")
        feature_df = prediction_row[FUNDAMENTAL_FEATURES].fillna(0)
        model_to_use = fundamental_model

    outcome_pred = model_to_use.predict(feature_df)[0]
    match_winner = {2: "Home Win", 1: "Draw", 0: "Away Win"}[outcome_pred]
    
    fundamental_feature_df = feature_df[FUNDAMENTAL_FEATURES]
    ou_pred = ou_model.predict(fundamental_feature_df)[0]
    over_under = "Over 2.5 Goals" if ou_pred == 1 else "Under 2.5 Goals"
    
    btts_pred = btts_model.predict(fundamental_feature_df)[0]
    btts = "Yes" if btts_pred == 1 else "No"
    
    predictions = {"Match Winner": match_winner, "Over/Under 2.5": over_under, "Both Teams to Score": btts}
    output_text = f"Predictions for: {home_team} vs. {away_team}\n" + "-"*40
    for key, value in predictions.items():
        output_text += f"\n- {key}: {value}"
    return output_text

# --- The Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚽ Football Match Predictor by AB77")
    gr.Markdown("Select teams for a fundamentals-based prediction, or add live odds for a more powerful, market-aware prediction.")
    
    with gr.Row():
        league_dd = gr.Dropdown(choices=sorted(teams_by_league.keys()), label="Select League")
    
    with gr.Row():
        # Components for standard league selection
        home_dd = gr.Dropdown(label="Home Team", visible=True)
        away_dd = gr.Dropdown(label="Away Team", visible=True)
        
        # Components for "Other/International" selection with autocomplete
        home_search_dd = gr.Dropdown(label="Search Home Team", choices=all_teams, filterable=True, visible=False)
        away_search_dd = gr.Dropdown(label="Search Away Team", choices=all_teams, filterable=True, visible=False)

    gr.Markdown("**(Optional) Enter Live Odds for a Market-Aware Prediction:**")
    with gr.Row():
        odds_h_in = gr.Number(label="Home Win Odds (e.g., 2.50)")
        odds_d_in = gr.Number(label="Draw Odds (e.g., 3.10)")
        odds_a_in = gr.Number(label="Away Win Odds (e.g., 2.80)")
        
    predict_btn = gr.Button("Generate Prediction", variant="primary")
    output_txt = gr.Textbox(label="Model Predictions", lines=4)

    def on_league_change(league):
        if league == "Other/International":
            return (
                gr.update(visible=False), # Hide standard home_dd
                gr.update(visible=False), # Hide standard away_dd
                gr.update(visible=True),  # Show searchable home_search_dd
                gr.update(visible=True)   # Show searchable away_search_dd
            )
        else:
            teams = teams_by_league.get(league, [])
            return (
                gr.update(visible=True, choices=teams, value=teams[0] if teams else None), # Show and update standard home_dd
                gr.update(visible=True, choices=teams, value=teams[1] if len(teams) > 1 else None), # Show and update standard away_dd
                gr.update(visible=False), # Hide searchable home_search_dd
                gr.update(visible=False)  # Hide searchable away_search_dd
            )

    def combined_predict(league, home_dd_val, away_dd_val, home_search_val, away_search_val, odds_h, odds_d, odds_a):
        if league == "Other/International":
            home_team = home_search_val
            away_team = away_search_val
        else:
            home_team = home_dd_val
            away_team = away_dd_val
        return predict_match_hybrid(home_team, away_team, odds_h, odds_d, odds_a)

    league_dd.change(fn=on_league_change, inputs=league_dd, outputs=[home_dd, away_dd, home_search_dd, away_search_dd])
    
    predict_btn.click(
        fn=combined_predict, 
        inputs=[league_dd, home_dd, away_dd, home_search_dd, away_search_dd, odds_h_in, odds_d_in, odds_a_in], 
        outputs=[output_txt]
    )
    
print("🚀 Launching the final hybrid Gradio application...")
demo.launch()