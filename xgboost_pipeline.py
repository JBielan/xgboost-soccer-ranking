import pandas as pd

# Read the dataset
data = pd.read_csv('data.csv')

# Choose interesting columns
data = data.loc[:, ['Age', 'Nationality', 'Value',
                    'Wage', 'Preferred Foot', 'Overall']]

# Age feature transformation
bins = [10, 20, 30, 40, 50]
group_names = ['10-19', '20-29', '30-39', '40-49']
age_categories = pd.cut(data['Age'], bins, labels=group_names)
data['Age'] = age_categories

# Categorical features into one-hot-encoding
one_hot_cols = ['Age', 'Nationality', 'Preferred Foot']
data = pd.get_dummies(data, columns=one_hot_cols)


# Value and Wage transformation function
def money_to_number(value_str):
    value_str = value_str.replace('€', "")

    if 'M' in value_str:
        value_str = value_str.replace('M', "")
        value = float(value_str)*1000000
    elif 'K' in value_str:
        value_str = value_str.replace('K', "")
        value = float(value_str)*1000
    else:
        value = float(value_str)

    return value


# Value and Wage transformation
data['Value'] = data['Value'].apply(money_to_number)
data['Wage'] = data['Wage'].apply(money_to_number)

# Splitting to training and testing data
import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(data.drop(['Overall'], axis=1))
y = np.array(data['Overall'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Preparing data for XGBoost training
import xgboost as xgb
D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)

# Defining an XGBoost model
param = {
    'eta': 0.3,
    'max_depth': 6,
    'objective': 'reg:squarederror'}
steps = 20

# Training the model
model = xgb.train(param, D_train, steps)

# Testing the model
from sklearn.metrics import mean_squared_error
preds = model.predict(D_test)
error = mean_squared_error(y_test, preds)
print("Squared error: ", error)

# Preparing new records with zeros in every column
ronaldo = data.iloc[0] * 0
messi = data.iloc[1] * 0

# Fulfilling particular columns for players
ronaldo['Value', 'Wage', 'Age_30-39', 'Nationality_Portugal', 'Preferred Foot_Right'] = [60000000, 3100000, 1, 1, 1]
messi['Value', 'Wage', 'Age_30-39', 'Nationality_Portugal', 'Preferred Foot_Left'] = [120000000, 2900000, 1, 1, 1]

# Preparing XGBoost format for prediction
ronaldo = xgb.DMatrix(ronaldo.drop(['Overall']))
messi = xgb.DMatrix(messi.drop(['Overall']))

# Predicting rankings
print('Cristiano Ronaldo ranking: {}'.format(model.predict(ronaldo)))
print('Lionel Messi ranking: {}'.format(model.predict(messi)))





















