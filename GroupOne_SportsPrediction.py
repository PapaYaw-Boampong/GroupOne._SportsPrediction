# -*- coding: utf-8 -*-
"""GroupOne_SportsPrediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KPs5c7COe1avEXe6YR6_a2YyXXfgPuWL
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold,GridSearchCV

pd.set_option('display.max_rows', None)

from google.colab import drive
drive.mount('/content/drive')

data_2021= pd.read_csv("/content/drive/My Drive/Colab Notebooks/players_21.csv")
data_2022 = pd.read_csv("/content/drive/My Drive/Colab Notebooks//players_22.csv", dtype={25: 'str', 108: 'str'})

# Display 2021 data
data_2021.head()

# Display 2022 data
data_2022.head()

to_drop =['sofifa_id', 'player_url', 'short_name', 'long_name','dob',
          'club_team_id','nationality_id','nation_team_id',
          'real_face','player_tags','player_traits',
          'club_logo_url','club_flag_url', 'nation_logo_url',
          'nation_flag_url','player_face_url',"nation_position","club_loaned_from"]

data_2021.drop(to_drop, axis = 1 , inplace =True )
data_2022.drop(to_drop, axis = 1 , inplace =True )

data_2021.dtypes

# extracting first number from the string
def extract_first_number(s):
    s = str(s)
    if "+" in s:
      return int(s.split('+')[0].strip())
    else:
      return int(s.split('-')[0].strip())


cols_to_modify = ['ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram',
                  'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb',
                  'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk']

for col in cols_to_modify:
    data_2021[ col] = data_2021[col].apply( extract_first_number)

for col in cols_to_modify:
    data_2022[ col] = data_2022[col].apply( extract_first_number)

# # Group by the last letter of the position columns using mean
# mean_by_position = data_2021[cols_to_modify].groupby(data_2021[cols_to_modify].columns.str[-1], axis=1).mean()

# data_2021 = data_2021.drop(columns=cols_to_modify)
# data_2021 = pd.concat([data_2021, mean_by_position], axis=1)


# mean_by_position2 = data_2022[cols_to_modify].groupby(data_2022[cols_to_modify].columns.str[-1], axis=1).mean()

# data_2022 = data_2022.drop(columns=cols_to_modify)
# data_2022 = pd.concat([data_2022, mean_by_position], axis=1)

# Defining position groups
striker_positions = ['ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw']
midfielder_positions = ['lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb']
defender_positions = ['lb', 'lcb', 'cb', 'rcb', 'rb']
goalkeeper_position = ['gk']

# Computing means and creating new columns
data_2021['Striker_Avg'] = data_2021[striker_positions].mean(axis=1)
data_2021['Midfielder_Avg'] = data_2021[midfielder_positions].mean(axis=1)
data_2021['Defender_Avg'] = data_2021[defender_positions].mean(axis=1)
data_2021['Goalkeeper'] = data_2021[goalkeeper_position]  # Directly assigning as there's only one column for goalkeepers



data_2022['Striker_Avg'] = data_2022[striker_positions].mean(axis=1)
data_2022['Midfielder_Avg'] = data_2021[midfielder_positions].mean(axis=1)
data_2021['Defender_Avg'] = data_2022[defender_positions].mean(axis=1)
data_2022['Goalkeeper'] = data_2022[goalkeeper_position]  # Directly assigning as there's only one column for goalkeepers

# Drop original columns
data_2021 = data_2022.drop(striker_positions + midfielder_positions + defender_positions + goalkeeper_position, axis=1)
data_2022 = data_2022.drop(striker_positions + midfielder_positions + defender_positions + goalkeeper_position, axis=1)

data_2021.dtypes

import numpy as np

# 2021 categorical and numeric data
categorical_features_2021 = data_2021.select_dtypes(exclude=[np.number])
numeric_features_2021 = data_2021.select_dtypes(include=[np.number])


# 2022 categorical and numeric data
categorical_features_2022 = data_2022.select_dtypes(exclude=[np.number])
numeric_features_2022 = data_2022.select_dtypes(include=[np.number])

categorical_features_2021.info()

"""Imputation and Scaling of numerical values

"""

categorical_features_2022.info()

numeric_imp=SimpleImputer(strategy='mean')

numeric_imp.fit(numeric_features_2021)

imputed_data = numeric_imp.transform(numeric_features_2021)
numeric_features_2021 =pd.DataFrame(imputed_data, columns=numeric_features_2021.columns)


imputed_data2 = numeric_imp.transform(numeric_features_2022)
numeric_features_2022 =pd.DataFrame(imputed_data2, columns=numeric_features_2021.columns)

y_train= numeric_features_2021['overall']
numeric_features_2021.drop("overall", axis = 1 , inplace =True)

y_test = numeric_features_2022['overall']
numeric_features_2022.drop("overall", axis = 1 , inplace =True)

"""Encoding and Imputing Categorical variables"""

encoders_2021 = {}  # to store our encoders for potential use later
encoders_2022 = {}
# Encoding Categorical features
imputer = SimpleImputer(strategy='most_frequent')
categorical_imputed_21 = imputer.fit_transform(categorical_features_2021)
categorical_imputed_22= imputer.transform(categorical_features_2022)

categorical_features_2021 = pd.DataFrame(categorical_imputed_21,columns = categorical_features_2021.columns)
categorical_features_2022 = pd.DataFrame(categorical_imputed_22,columns = categorical_features_2022.columns)


imputer = SimpleImputer(strategy='most_frequent')

for column in categorical_features_2021.columns:
    encoder = LabelEncoder()
    categorical_features_2021[column] = encoder.fit_transform(categorical_features_2021[column])
    encoders_2021[column] = encoder  # store the encoder

for column in categorical_features_2022.columns:
    encoder = LabelEncoder()
    categorical_features_2022[column] = encoder.fit_transform(categorical_features_2022[column])
    encoders_2022[column] = encoder  # store the encoder

temp_df_2021 = pd.concat([numeric_features_2021,categorical_features_2021, y_train], axis = 1)

temp_df_2022 = pd.concat([numeric_features_2022,categorical_features_2022,y_test], axis = 1)

"""**Correlation Analysis (EDA)**"""

correlation_matrix_2021 = temp_df_2021.corr()

highest_correlation = []

correlation_matrix_2021 = temp_df_2021.corr()
for (i,x) in correlation_matrix_2021["overall"].items():
  if x > 0.45:
    highest_correlation.append(i)

highest_correlation

train_data = temp_df_2021.copy()[highest_correlation]
df_2022 = temp_df_2022.copy()[highest_correlation]

train_data.dtypes

import matplotlib.pyplot as plt
import seaborn as sns


for i in train_data.columns:
  sns.scatterplot(data=train_data, x='overall', y = i)
  plt.show()

# Training data
y = train_data['overall']
train_data.drop("overall", axis = 1 , inplace =True)


# 2022 data
y_test_2022 = df_2022['overall']
df_2022.drop("overall", axis = 1 , inplace =True)

#Scaling numeric values
scaler = StandardScaler()
scaler.fit(train_data)

scaled_data_2021 = scaler.transform(train_data)

scaled_data_2022 = scaler.transform(df_2022)


train_data = pd.DataFrame(scaled_data_2021,columns = train_data.columns)
df_2022 = pd.DataFrame(scaled_data_2022,columns = df_2022.columns)

X = train_data

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.2,random_state=42)

"""**Random Forest Regressor**"""

rf = RandomForestRegressor(n_estimators=120, random_state=42)
rf.fit(Xtrain, Ytrain)

y_pred = rf.predict(Xtest)

mse = mean_absolute_error (y_pred, Ytest)
print(f"Mean Absolute Error: {mse}")

"""**Gradient Booster**

"""

gbrt = GradientBoostingRegressor(n_estimators=120, learning_rate=0.01)
gbrt.fit(Xtrain, Ytrain)

y_pred=gbrt.predict(Xtest)

mse = mean_absolute_error (y_pred, Ytest)
print(f"Mean Absolute Error: {mse}")

"""**GX Booster**"""

gbr = XGBRegressor(n_estimators=120, learning_rate=0.01)
gbr.fit(Xtrain, Ytrain)

y_pred=gbr.predict(Xtest)
mse = mean_absolute_error (y_pred, Ytest)
print(f"Mean Absolute Error: {mse}")

"""

**Cross Validation for random forest regressor**

"""

cv=KFold(n_splits=3)

PARAMETERS = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [3, 5, 10, 20],
}

rfr = RandomForestRegressor()
model_rfr = GridSearchCV(rfr,param_grid=PARAMETERS,cv=cv,scoring="neg_mean_absolute_error")
model_rfr.fit(Xtrain,Ytrain)

y_pred=model_rfr.predict(Xtest)

mse = mean_absolute_error (y_pred, Ytest)
print(f"Mean Absolute Error: {mse}")

model_rfr.best_estimator_

"""**Cross Validation for GradientBoostingRegressor**"""

cv2=KFold(n_splits=3)
PARAMETERS = {
    'n_estimators': [10, 50, 100, 120 ,200, 500, 1000],
    'max_depth': [3, 5, 7, 10, 20],
    "learning_rate" : [0.03, 0.1, 0.3,0.01]
}

gbrt = GradientBoostingRegressor()

model_gbrt = GridSearchCV(gbrt,param_grid=PARAMETERS,cv=cv,scoring="neg_mean_absolute_error")
model_gbrt.fit(Xtrain,Ytrain)

y_pred=model_gbrt.predict(Xtest)

mse = mean_absolute_error (y_pred, Ytest)
print(f"Mean Absolute Error: {mse}")

model_gbrt.best_estimator_

"""**Cross Validation for XGBRegressor**"""

cv3=KFold(n_splits=3)

xgbr = XGBRegressor()

model_xgbr = GridSearchCV(xgbr,param_grid=PARAMETERS,cv=cv3,scoring="neg_mean_absolute_error")
model_xgbr.fit(Xtrain,Ytrain)

y_pred=model_xgbr.predict(Xtest)

mse = mean_absolute_error (y_pred, Ytest)
print(f"Mean Absolute Error: {mse}")

model_xgbr.best_estimator_

"""**Best fine Tuned Model**"""

gbr_fine_tuned = GradientBoostingRegressor(learning_rate=0.03, max_depth=10, n_estimators=1000)

gbr_fine_tuned.fit(Xtrain, Ytrain)

#learning_rate=0.03, max_depth=10, n_estimators=1000

y_pred = gbr_fine_tuned.predict(Xtest)

mse = mean_absolute_error (y_pred, Ytest)
print(f"Mean Absolute Error: {mse}")

"""**Testing on 2022**"""

y_pred=gbr_fine_tuned.predict(df_2022)

mse = mean_absolute_error (y_pred, y_test_2022)
print(f"Mean Absolute Error: {mse}")

"""**Pickeling best model**"""

model_data = {
    "model" : gbr_fine_tuned,
    "scaler" : scaler
}

import pickle

# Save to file
with open("fifa_ml.pkl", "wb") as content:
    pickle.dump(model_data, content)

# !pip install --upgrade scikit-learn