import streamlit as st
import pickle
import pandas as pd
import sklearn
from sklearn.utils import resample
import numpy as np

# Assume x_single is your single row of data to predict on
# x_single = ...

n_bootstrap = 1000
bootstrap_predictions = []

for _ in range(n_bootstrap):
    X_resampled, y_resampled = resample(X_train, y_train)
    model = GradientBoostingRegressor()
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict([x_single])
    bootstrap_predictions.append(y_pred)

def run():
    st.set_page_config(
        page_title="Fifa Player Overall Rating Prediction",
        page_icon="⚽",
    )
    model = pickle.load(open('fifa_ml.pkl', "rb"))["model"]
    scaler = pickle.load(open('fifa_ml.pkl', "rb"))["scaler"]
    
    xtrain = pickle.load(open('train_data.pkl', "rb"))
    ytrain = pickle.load(open('test_data.pkl', "rb"))
    
  
    st.write("# Group One Fifa Player Overall Rating Prediction")
    st.write("\n Enter data (1-100)")

    # User input
    def user_input():
        potential = st.number_input("Potential (1-100)", value=80, min_value=1, max_value=100)
        value_eur = st.number_input("Value (in EUR)", value=100000, min_value=1)
        wage_eur = st.number_input("Wage (in EUR)", value=1000, min_value=1)
        age = st.number_input("Age", value=20, min_value=18, max_value=70)
        int_rep = st.number_input("International Reputation", min_value=1, max_value=5, value=3)
        release_clause_eur = st.number_input("Release Clause (in EUR)", value=100000, min_value=1)
        shooting = st.number_input("Shooting (1-100)", value=50, min_value=1, max_value=100)
        passing = st.number_input("Passing (1-100)", value=50, min_value=1, max_value=100)
        dribbling = st.number_input("Dribbling (1-100)", value=50, min_value=1, max_value=100)
        physic = st.number_input("Physic (1-100)", value=50, min_value=1, max_value=100)
        attacking_short_passing = st.number_input("Attacking Short Passing (1-100)", value=50, min_value=1, max_value=100)
        skill_long_passing = st.number_input("Skill Long Passing (1-100)", value=50, min_value=1, max_value=100)
        skill_ball_control = st.number_input("Skill Ball Control (1-100)", value=50, min_value=1, max_value=100)
        movement_reactions = st.number_input("Movement Reactions (1-100)", value=50, min_value=1, max_value=100)
        power_shot_power = st.number_input("Power Shot Power (1-100)", value=50, min_value=1, max_value=100)
        mentality_vision = st.number_input("Mentality Vision (1-100)", value=50, min_value=1, max_value=100)
        mentality_composure = st.number_input("Mentality Composure (1-100)", value=50, min_value=1, max_value=100)
        Striker_Avg = st.number_input("Striker Average (1-100)", value=50, min_value=1, max_value=100)
        Midfielder_Avg = st.number_input("Midfielder Average (1-100)", value=50, min_value=1, max_value=100)

        d = {
            'potential': potential,
            'value_eur': value_eur,
            'wage_eur': wage_eur,
            'age': age,
            'international_reputation': int_rep,
            'release_clause_eur': release_clause_eur,
            'shooting': shooting,
            'passing': passing,
            'dribbling': dribbling,
            'physic': physic,
            'attacking_short_passing': attacking_short_passing,
            'skill_long_passing': skill_long_passing,
            'skill_ball_control': skill_ball_control,
            'movement_reactions': movement_reactions,
            'power_shot_power': power_shot_power,
            'mentality_vision': mentality_vision,
            'mentality_composure': mentality_composure,
            'Striker_Avg': Striker_Avg,
            'Midfielder_Avg': Midfielder_Avg
        }
        a = pd.DataFrame(d, index=[0])
        scaled_data = scaler.transform(a)
        return pd.DataFrame(scaled_data, columns=a.columns, index=[0])

    data = user_input()

    # Function to pass user input to the model
    def predict_rating(data_input,x,y):
        Xresampled, Yresampled = resample(x, y)
        model.fit(Xresampled, Yresampled)
        return model.predict(data_input)

    predict_button = st.button("Predict Player Rating")
    if predict_button:
        n_bootstrap = 1000
        predictions = []
        
        for i in range(n_bootstrap):
          y_pred  = predict_rating(data,xtrain,ytrain)
          predictions.append(y_pred)
        
        bootstrap_predictions = np.array(predictions)
        # Calculate mean and standard deviation of predictions
        mean_prediction = bootstrap_predictions.mean()
        std_prediction = bootstrap_predictions.std()

        z_score = 1.96
        lower_bound = mean_prediction - z_score * std_prediction
        upper_bound = mean_prediction + z_score * std_prediction


        prediction = mean_prediction
        st.write(f"Predicted Overall Player Rating: {prediction[0]:.2f}")
        st.write(f"95% Prediction Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")
    # Button to make predictions


if __name__ == "__main__":
    run()
