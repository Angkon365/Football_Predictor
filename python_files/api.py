import pandas as pd
import numpy as np
import xgboost as xgb
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Constants and Feature Lists ---
PROJECT_FOLDER = r'C:\Users\angkon\Desktop\Project'
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'Data')
DF_PATH = os.path.join(PROJECT_FOLDER, 'final_model_dataset_h2h.pkl')
COMPETITIONS_PATH = os.path.join(DATA_FOLDER, 'competitions.csv')

# --- CORRECTED FEATURE LISTS ---
# 'poisson_xg_diff' has been removed from this list. This was the cause of the error.
BASE_FEATURES = [
    'elo_diff', 'coach_elo_diff', 'market_value_diff', 'market_value_ratio',
    'prob_H', 'prob_D', 'prob_A', 'starting_xi_value_diff',
    'formation_attack_diff', 'clv_home', 'clv_draw', 'clv_away',
    'home_cs_form', 'home_fts_form', 'away_cs_form', 'away_fts_form'
]
FUNDAMENTAL_FEATURES = [
    'elo_diff', 'coach_elo_diff', 'market_value_diff', 'market_value_ratio',
    'starting_xi_value_diff', 'formation_attack_diff', 'home_cs_form', 'home_fts_form',
    'away_cs_form', 'away_fts_form', 'h2h_home_win_pct', 'h2h_home_draw_pct',
    'h2h_home_avg_gs', 'h2h_home_avg_gc'
]
FULL_FEATURES = FUNDAMENTAL_FEATURES + ['prob_H', 'prob_D', 'prob_A', 'clv_home', 'clv_draw', 'clv_away']

# --- Global Variables ---
df_master, fundamental_model, full_feature_model, final_ou_model, final_btts_model, xg_home_model, xg_away_model, teams_by_league, all_teams = (None,) * 9

def setup_and_train_all_models():
    """
    This is the core logic from your working Gradio app. It runs once when the server starts.
    """
    global df_master, fundamental_model, full_feature_model, final_ou_model, final_btts_model, xg_home_model, xg_away_model, teams_by_league, all_teams
    
    print("--- API Server Initializing: Loading data and training models... ---")
    df = pd.read_pickle(DF_PATH)
    df.sort_values(by='date', inplace=True)
    df['market_value_ratio'] = df['total_market_value_home'] / (df['total_market_value_away'] + 1)
    
    for col in ['home_cs_form', 'home_fts_form', 'away_cs_form', 'away_fts_form']:
        if col not in df.columns: df[col] = 0.0

    X_base = df[[col for col in BASE_FEATURES if col in df.columns]].fillna(0)
    y_btts_target = ((df['home_club_goals'] > 0) & (df['away_club_goals'] > 0)).astype(int)
    y_ou_target = (df['total_goals'] > 2.5).astype(int)
    y_outcome_target = df['result_encoded']

    xg_home_model = xgb.XGBRegressor(objective='count:poisson', random_state=42).fit(X_base, df['home_club_goals'])
    xg_away_model = xgb.XGBRegressor(objective='count:poisson', random_state=42).fit(X_base, df['away_club_goals'])

    df['stacked_xg_home'] = xg_home_model.predict(X_base)
    df['stacked_xg_away'] = xg_away_model.predict(X_base)
    df['stacked_xg_total'] = df['stacked_xg_home'] + df['stacked_xg_away']
    
    X_stacked = df[X_base.columns.tolist() + ['stacked_xg_home', 'stacked_xg_away', 'stacked_xg_total']].fillna(0)
    split_index = int(len(df) * 0.80)
    
    y_btts_train = y_btts_target.iloc[:split_index]
    scale_pos_weight_btts = y_btts_train.value_counts()[0] / y_btts_train.value_counts()[1]
    final_btts_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', scale_pos_weight=scale_pos_weight_btts, random_state=42).fit(X_stacked.iloc[:split_index], y_btts_train)

    y_ou_train = y_ou_target.iloc[:split_index]
    scale_pos_weight_ou = y_ou_train.value_counts()[0] / y_ou_train.value_counts()[1]
    final_ou_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', scale_pos_weight=scale_pos_weight_ou, random_state=42).fit(X_stacked.iloc[:split_index], y_ou_train)
    
    y_outcome_train = y_outcome_target.iloc[:split_index]
    sample_weights = np.ones(len(y_outcome_train)); sample_weights[y_outcome_train == 1] *= 1.8
    
    X_fundamental = df[[col for col in FUNDAMENTAL_FEATURES if col in df.columns]].fillna(0)
    fundamental_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42).fit(X_fundamental.iloc[:split_index], y_outcome_train, sample_weight=sample_weights)

    X_full = df[[col for col in FULL_FEATURES if col in df.columns]].fillna(0)
    full_feature_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42).fit(X_full.iloc[:split_index], y_outcome_train, sample_weight=sample_weights)

    df_competitions = pd.read_csv(COMPETITIONS_PATH)
    league_name_map = df_competitions.set_index('competition_id')['name'].to_dict()
    teams_by_league = {league_name_map.get(comp_id, comp_id): sorted(pd.concat([df[df['competition_id'] == comp_id]['home_team'], df[df['competition_id'] == comp_id]['away_team']]).dropna().unique()) for comp_id in df['competition_id'].unique()}
    teams_by_league['Other/International'] = []
    all_teams = sorted(pd.concat([df['home_team'], df['away_team']]).dropna().unique())
    df_master = df

def predict_match_hybrid(home_team, away_team, odds_h, odds_d, odds_a):
    if not home_team or not away_team: return {"error": "Please provide both teams."}
    if home_team == away_team: return {"error": "Teams cannot be the same."}
    if home_team not in all_teams or away_team not in all_teams: return {"error": "One or both teams not found."}
    
    prediction_row = pd.DataFrame(index=[0])
    home_stats = df_master[(df_master['home_team'] == home_team) | (df_master['away_team'] == home_team)].sort_values(by='date').iloc[-1]
    away_stats = df_master[(df_master['home_team'] == away_team) | (df_master['away_team'] == away_team)].sort_values(by='date').iloc[-1]
    def get_stat(df_row, team_name, home_stat_name, away_stat_name): return df_row[home_stat_name] if df_row['home_team'] == team_name else df_row[away_stat_name]
    prediction_row['elo_diff'] = get_stat(home_stats, home_team, 'elo_home', 'elo_away') - get_stat(away_stats, away_team, 'elo_away', 'elo_home')
    prediction_row['coach_elo_diff'] = get_stat(home_stats, home_team, 'coach_elo_home', 'coach_elo_away') - get_stat(away_stats, away_team, 'coach_elo_away', 'coach_elo_home')
    home_market_value, away_market_value = get_stat(home_stats, home_team, 'total_market_value_home', 'total_market_value_away'), get_stat(away_stats, away_team, 'total_market_value_away', 'total_market_value_home')
    prediction_row['market_value_diff'], prediction_row['market_value_ratio'] = home_market_value - away_market_value, home_market_value / (away_market_value + 1)
    prediction_row['starting_xi_value_diff'] = get_stat(home_stats, home_team, 'starting_xi_value_home', 'starting_xi_value_away') - get_stat(away_stats, away_team, 'starting_xi_value_away', 'starting_xi_value_home')
    prediction_row['formation_attack_diff'] = get_stat(home_stats, home_team, 'formation_attack_score_home', 'formation_attack_score_away') - get_stat(away_stats, away_team, 'formation_attack_score_away', 'formation_attack_score_home')
    h2h_stats_df = df_master[((df_master['home_team'] == home_team) & (df_master['away_team'] == away_team))].sort_values(by='date')
    if not h2h_stats_df.empty:
        latest_h2h = h2h_stats_df.iloc[-1]
        for col in ['home_cs_form', 'home_fts_form', 'h2h_home_win_pct', 'h2h_home_draw_pct', 'h2h_home_avg_gs', 'h2h_home_avg_gc', 'away_cs_form', 'away_fts_form']: prediction_row[col] = latest_h2h[col]
    else: prediction_row[['home_cs_form', 'home_fts_form', 'away_cs_form', 'away_fts_form', 'h2h_home_win_pct', 'h2h_home_draw_pct', 'h2h_home_avg_gs', 'h2h_home_avg_gc']] = 0.0

    if all(x is not None and x > 0 for x in [odds_h, odds_d, odds_a]):
        prediction_row['prob_H'], prediction_row['prob_D'], prediction_row['prob_A'] = 1/odds_h, 1/odds_d, 1/odds_a
        prediction_row['clv_home'], prediction_row['clv_draw'], prediction_row['clv_away'] = (prediction_row['prob_H']*odds_h)-1, (prediction_row['prob_D']*odds_d)-1, (prediction_row['prob_A']*odds_a)-1
        outcome_pred = full_feature_model.predict(prediction_row[FULL_FEATURES].fillna(0))[0]
    else: outcome_pred = fundamental_model.predict(prediction_row[FUNDAMENTAL_FEATURES].fillna(0))[0]
    match_winner = {0: "Away Win", 1: "Draw", 2: "Home Win"}[outcome_pred]
    
    # This section now works correctly because 'poisson_xg_diff' is no longer required here.
    base_feature_df = prediction_row[BASE_FEATURES].fillna(0)
    prediction_row['stacked_xg_home'], prediction_row['stacked_xg_away'] = xg_home_model.predict(base_feature_df)[0], xg_away_model.predict(base_feature_df)[0]
    prediction_row['stacked_xg_total'] = prediction_row['stacked_xg_home'] + prediction_row['stacked_xg_away']
    stacked_feature_df = prediction_row[base_feature_df.columns.tolist() + ['stacked_xg_home', 'stacked_xg_away', 'stacked_xg_total']].fillna(0)
    over_under = "Over 2.5 Goals" if final_ou_model.predict(stacked_feature_df)[0] == 1 else "Under 2.5 Goals"
    btts = "Yes" if final_btts_model.predict(stacked_feature_df)[0] == 1 else "No"
    return {"matchWinner": match_winner, "overUnder": over_under, "btts": btts}

@app.route('/teams', methods=['GET'])
def get_teams(): return jsonify(teams_by_league if teams_by_league else {"error": "Data not ready."})

@app.route('/predict', methods=['POST'])
def handle_prediction():
    try:
        data = request.get_json()
        result = predict_match_hybrid(data.get('home_team'), data.get('away_team'), data.get('odds_h'), data.get('odds_d'), data.get('odds_a'))
        return jsonify(result)
    except Exception as e:
        # We will now print the full error for better debugging
        import traceback
        print(f"ERROR during prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    try:
        setup_and_train_all_models()
        print("\n✅ SUCCESS: Model setup complete. Starting Flask server...")
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        import traceback
        print(f"\n❌ FATAL ERROR: Failed to start the server. Error during setup: {e}")
        traceback.print_exc()