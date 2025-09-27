import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Football Predictor EDA Dashboard",
    page_icon="⚽",
    layout="wide"
)

# --- Title and Introduction ---
st.title("⚽ Football Match Predictor: EDA Dashboard")
st.markdown("""
This dashboard provides an interactive view of the data used to train the football match prediction model.
Explore the dataset, key visualizations from the Exploratory Data Analysis (EDA), and the most important features identified by the model.
""")

# --- Caching Data Loading ---
@st.cache_data
def load_data():
    """Loads the final dataset and the trained model, caching for performance."""
    data_path = 'aggregated_raw_odds_expanded.csv'
    # CORRECTED: Pointing to the file confirmed in your screenshot.
    model_path = 'final_model_dataset_h2h.pkl'

    if not os.path.exists(data_path):
        st.error(f"Error: Data file not found at '{data_path}'. Please ensure the CSV file is in the correct location.")
        return None, None
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at '{model_path}'. This should not happen if the file is in the same directory.")
        return None, None

    df = pd.read_csv(data_path)
    # This .pkl file contains the model pipeline
    model = joblib.load(model_path)
    return df, model

# --- Load Data ---
df, model = load_data()

# --- Main Dashboard Logic (only runs if data is loaded) ---
if df is not None and model is not None:

    # --- Section 1: Raw Data Explorer ---
    st.header("Data Explorer")
    st.markdown("A glimpse into the final dataset used for training the model.")
    st.dataframe(df.head(10))

    # --- Section 2: Key Visualizations ---
    st.header("Key Visualizations from EDA")

    col1, col2 = st.columns(2)

    with col1:
        # Bar Chart: Distribution of Match Outcomes
        st.subheader("Distribution of Match Outcomes")
        outcome_counts = df['result'].value_counts()
        fig_outcomes = px.bar(
            outcome_counts,
            x=outcome_counts.index,
            y=outcome_counts.values,
            labels={'x': 'Match Result', 'y': 'Number of Matches'},
            color=outcome_counts.index,
            color_discrete_map={'H': 'skyblue', 'D': 'lightgreen', 'A': 'salmon'},
            title="Home Win (H) vs. Draw (D) vs. Away Win (A)"
        )
        st.plotly_chart(fig_outcomes, use_container_width=True)

    with col2:
        # Histogram: Goals Scored
        st.subheader("Distribution of Goals Scored")
        fig_goals = px.histogram(
            df,
            x=['home_team_goal_count', 'away_team_goal_count'],
            barmode='overlay',
            marginal='box',
            labels={'value': 'Number of Goals'},
            title="Goals per Match by Home and Away Teams"
        )
        fig_goals.update_layout(xaxis_title="Goals")
        st.plotly_chart(fig_goals, use_container_width=True)

    # --- Section 3: Top Predictive Features ---
    st.header("Top Features Driving Predictions")
    st.markdown("The Random Forest model calculates the importance of each feature. Here are the top 15 features that have the most influence on predicting a match's outcome.")

    # Extracting feature importances from the model pipeline
    try:
        # Assuming the model is the last step in a pipeline that has a 'feature_importances_' attribute
        importances = model.named_steps['classifier'].feature_importances_
        
        # Get feature names from the preprocessing step of the pipeline
        preprocessor = model.named_steps['preprocessor']
        numeric_features = preprocessor.transformers_[0][2]
        categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        
        feature_names = list(numeric_features) + list(categorical_features)

        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(15)

        # Create Bar Chart
        fig_importance = px.bar(
            feature_importance_df.sort_values('importance', ascending=True),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 Most Important Features'
        )
        fig_importance.update_layout(yaxis_title="Feature", xaxis_title="Importance Score")
        st.plotly_chart(fig_importance, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not automatically extract feature importances from the loaded file. The file '{model_path}' might be a dataset instead of a model pipeline. Error: {e}")
        st.info("Ensure the '.pkl' file is a scikit-learn pipeline with named steps 'preprocessor' and 'classifier' for this feature to work.")

    # --- Section 4: Static EDA Snapshots from Notebook ---
    st.header("Static EDA Snapshots from Notebook")
    st.markdown("These are static images from the original analysis, providing additional context.")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if os.path.exists("features.png"):
            st.image("features.png", caption="Initial Feature Importance Plot")
        if os.path.exists("SHAP Bar Plot.png"):
            st.image("SHAP Bar Plot.png", caption="SHAP Bar Plot for Feature Impact")
            
    with col4:
        if os.path.exists("SHAP Summary Plot.png"):
            st.image("SHAP Summary Plot.png", caption="SHAP Summary Plot")
        if os.path.exists("draw.......png"):
            st.image("draw.......png", caption="Confusion Matrix for 'Draw' Class")

else:
    st.warning("Dashboard cannot be displayed because the required data or model files could not be loaded.")

