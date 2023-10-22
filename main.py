import streamlit as st
import pickle
import pandas as pd
import sklearn


def run():
    st.set_page_config(
        page_title="Fifa Player Overall Rating Prediction",
        page_icon="⚽",
    )
    model = pickle.load(open('fifa_ml.pkl', "rb"))["model"]
    scaler = pickle.load(open('fifa_ml.pkl', "rb"))["scaler"]
    # # Load the trained model from a file
    # with open('fifa_ml.pkl', 'rb') as file:
    #     model = pickle.load(file)['model']
    #     scaler = pickle.load(file)['scaler']

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
    def predict_rating(data_input):
        return model.predict(data_input)

    predict_button = st.button("Predict Player Rating")
    if predict_button:
        prediction = predict_rating(data)
        st.write(f"Predicted Overall Player Rating: {prediction[0]:.2f}")
    # Button to make predictions


run()
