# Football Match Result Predictor 
A machine learning data analysis project that predicts the outcome (Win, Draw, or Loss)and also Both teams to score (BTTS) and Over/Under 2.5 goals of football matches using historical data and a statistical model by leveraging detailed individual player performance data. Using a comprehensive dataset of player statistics from the top 5 European leagues from 2012 to 2025 and 6 other leagues, along with odds integration, the goal is to engineer team-level features to train a robust classification model. This repository contains the data processing, model training notebooks, and an interactive Gradio application intended for local use.

**Live Demo & Dashboard**
Project Dashboard (HTML): A static dashboard showcasing project visuals and data is live at the link below.

View the Live Dashboard Here

Interactive Predictor App (Gradio): The Gradio app is designed to run locally on your machine. Please follow the instructions in the "Installation and Usage" section below to run it.

** Table of Contents**
Project Overview

Data Source

Features

Technology Stack

Project Workflow

Getting Started: Running the App

File Structure

Contact

**Project Overview**
This project aims to solve the classification problem of predicting football match outcomes. By leveraging historical match data and team statistics, several machine learning models were trained and evaluated to find the most accurate predictor. The final model is served via an interactive web application, allowing users to select two teams and get an instant prediction for an upcoming match.

The primary business case is to provide a data-driven tool for fans, analysts, and betting enthusiasts to gain statistical insights into match probabilities, moving beyond simple intuition.

**Data Source**
Kaggle: Football Player Stats (2018-2023) (https://www.kaggle.com/datasets/davidcariboo/player-scores) (Raw Data)
https://www.football-data.co.uk/data.php (Odds Data)

Description: This dataset contains granular, player-level performance statistics for every match in the top 5 European leagues (England, Spain, Italy, Germany, and France) from the 2018-2019 season to the 2022-2023 season. Key features include goals, assists, shots, cards, and minutes played for each player in a given match.

**Features**
Data Collection: Gathers historical match data and betting odds.

Data Preprocessing: Cleans and transforms raw data, handling missing values and creating a structured dataset.

Feature Engineering: Creates meaningful features like team form, goal differentials, and head-to-head statistics.

Model Training: Implements and evaluates multiple classification models, including Logistic Regression, Random Forest, and XGBoost.

Interactive Prediction App: A user-friendly web interface built with Gradio where users can select teams to see the predicted outcome.

**Technology Stack**
Programming Language: Python

Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost

Web App Framework: Gradio

Development Environment: Jupyter Notebook

###  Project Workflow
The project was executed following a structured data science workflow:
1.  **Business Understanding:** Defined the project goals and success criteria as outlined in the Business Case Document.
2.  **Data Collection:** Acquired raw data from public APIs.
3.  **Exploratory Data Analysis (EDA):** Analyzed the data to uncover patterns, correlations, and initial insights.
4.  **Data Preprocessing & Feature Engineering:** Cleaned the dataset and engineered new features to improve model performance.
5.  **Model Building:** Trained several classification algorithms on the prepared data.
6.  **Model Evaluation:** Assessed models based on accuracy, precision, and recall to select the best-performing one.
7.  **Deployment:** Packaged the final model into a Gradio web application for easy access and interactivity.

---
**Getting Started: Running the App**
This project can be run locally or deployed to the web.

---

###  Installation and Usage

1. To run this project locally, please follow these steps:

a.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Angkon365/Football_Predictor.git](https://github.com/Angkon365/Football_Predictor.git)
    cd Football_Predictor
    ```

b.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

c.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

d.  **Run the Gradio application:**
    ```bash
    python predictor_app_hybrid.py
    ```
    Open your web browser and navigate to the local URL provided in the terminal (usually `http://127.0.0.1:7860`).

2. Deploying to Hugging Face Spaces (Optional)
This application is ready to be deployed directly to Hugging Face Spaces.

Create a Space: Sign up for a free Hugging Face account and create a new "Gradio" Space.

Link Repository: Choose the option to clone from a Git repository and provide the URL to this GitHub repo.

Deploy: Hugging Face will automatically handle the installation and launch the application, making it publicly available.    

---

###  File Structure  

├── models/                     # Saved model files (.json, .pkl)
├── notebooks/                  # Jupyter notebooks for EDA and model training
├── data/                       # Excel-generated files by notebook
├── python_files/               # All relevant Python files
├── predictor_app_hybrid.py     # Main Python script for the Gradio app
├── requirements.txt            # List of required Python packages
├── Team_Map_Manual.txt         # Manual team mapping txt file for reference
├── .gitignore                  # Files and folders to ignore
└── README.md                   # Project documentation

---

### 📬 Contact
Angkon Biswas - [LinkedIn](https://www.linkedin.com/in/angkon-biswas-a13167214/) - angkonbiswas1993@gmail.com

Project Link: [https://github.com/Angkon365/Football_Predictor](https://github.com/Angkon365/Football_Predictor)
