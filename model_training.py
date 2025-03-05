import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Loading training dataframe
df = pd.read_csv('train_v9rqX0R.csv')
print("Data shape:", df.shape)

#Loading test
test = pd.read_csv("test_AbJTz2l.csv")

# Missing value interpolation
# Outlet Size -> By outlet type
size_mode = df.groupby('Outlet_Type')['Outlet_Size'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
print(size_mode)

df['Outlet_Size'] = df.apply(
    lambda x: size_mode[x['Outlet_Type']] if pd.isnull(x['Outlet_Size']) else x['Outlet_Size'], axis=1
)
test['Outlet_Size'] = test.apply(
    lambda x: size_mode[x['Outlet_Type']] if pd.isnull(x['Outlet_Size']) else x['Outlet_Size'], axis=1
)

#Let's check for Item Weight now. Let's try imputing Item Weight by averaage weight of the same Item_Identifier
itemwise_avg_weight = df.groupby('Item_Identifier')['Item_Weight'].mean()

df['Item_Weight'] = df.apply(
    lambda x: itemwise_avg_weight[x['Item_Identifier']] if pd.isna(x['Item_Weight']) else x['Item_Weight'], axis=1)

test['Item_Weight'] = test.apply(
    lambda row: itemwise_avg_weight[row['Item_Identifier']] if pd.isnull(row['Item_Weight']) else row['Item_Weight'],
    axis=1)

# For remaining ones
overall_median_weight = df['Item_Weight'].median()
df['Item_Weight'].fillna(overall_median_weight, inplace=True)
test['Item_Weight'].fillna(overall_median_weight, inplace=True)

# Cleanup
#LF, low fat , Low Fat -> should be same.
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat','low fat': 'Low Fat','reg': 'Regular'})
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace({'LF': 'Low Fat','low fat': 'Low Fat','reg': 'Regular'})
df['Outlet_Age'] = 2013 - df['Outlet_Establishment_Year']
test['Outlet_Age'] = 2013 - test['Outlet_Establishment_Year']

# 0 visibility outlier handling: Replace zero visibilities with the mean visibility of that item
visibility_avg = df.pivot_table(values='Item_Visibility', index='Item_Identifier', aggfunc='mean')
zero_mask = (df['Item_Visibility'] == 0)
df.loc[zero_mask, 'Item_Visibility'] = df.loc[zero_mask, 'Item_Identifier'].apply(lambda x: visibility_avg.at[x, 'Item_Visibility'])
print("Number of zero visibility entries after imputation:", (df['Item_Visibility'] == 0).sum())

zero_mask = (test['Item_Visibility'] == 0)
test.loc[zero_mask, 'Item_Visibility'] = test.loc[zero_mask, 'Item_Identifier'].apply(lambda x: visibility_avg.at[x, 'Item_Visibility'])
print("Number of zero visibility entries after imputation:", (test['Item_Visibility'] == 0).sum())

# Item Identifier consumables, FD first 2 chars sufficient to classify.
df['Item_Identifier_Categories'] = df['Item_Identifier'].str[0:2] 
test['Item_Identifier_Categories']  = test['Item_Identifier'].str[0:2]

#renaming
train = df

# label encoding
train['Outlet_Size'] = train['Outlet_Size'].map({'Small'  : 1,
                                                 'Medium' : 2,
                                                 'High'   : 3
                                                 }).astype(int)

test['Outlet_Size'] = test['Outlet_Size'].map({'Small'  : 1,
                                                 'Medium' : 2,
                                                 'High'   : 3
                                                 }).astype(int)


#Tier 1 -> 1
train['Outlet_Location_Type'] = train['Outlet_Location_Type'].str[-1:].astype(int)
test['Outlet_Location_Type']  = test['Outlet_Location_Type'].str[-1:].astype(int)

encoder = LabelEncoder()
ordinal_features = ['Item_Fat_Content', 'Outlet_Type', 'Outlet_Location_Type']

for feature in ordinal_features:
    train[feature] = encoder.fit_transform(train[feature])
    test[feature]  = encoder.fit_transform(test[feature])

train = pd.get_dummies(train, columns=['Item_Type', 'Item_Identifier_Categories'], drop_first=True)
test  = pd.get_dummies(test,  columns=['Item_Type', 'Item_Identifier_Categories'], drop_first=True)

# Drop cols based on EDA and current relevancy.
train.drop(labels=['Item_Identifier', "Outlet_Establishment_Year", "Outlet_Identifier"], axis=1, inplace=True)
test.drop(labels=['Item_Identifier', "Outlet_Establishment_Year", "Outlet_Identifier"],  axis=1, inplace=True)

# Independent and dependent feature
X = train.drop('Item_Outlet_Sales', axis=1)
y = train['Item_Outlet_Sales']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.columns)

# Modelling

categorical_feats = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Type', 'Outlet_Location_Type']
numeric_feats = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']


preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feats)
])

xgb_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
])


# Parameter grid
param_grid_xgb = {
    'xgb__n_estimators': [50, 100, 150,200],
    'xgb__max_depth': [3, 6, 9, 12],  
    'xgb__learning_rate': [0.05, 0.1, 0.15, 0.2],  
    'xgb__min_child_weight': [1, 3, 5]
}

grid_xgb = GridSearchCV(xgb_pipeline, param_grid_xgb, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)

grid_xgb.fit(X_train, y_train)

print("Best XGB Params:", grid_xgb.best_params_)

"""
y_pred_xgb = grid_xgb.predict(X_test)
import math
xgb_rmse = math.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_mae  = mean_absolute_error(y_test, y_pred_xgb)
xgb_r2   = r2_score(y_test, y_pred_xgb)

xgb_r2   = r2_score(y_test, y_pred_xgb)
print(f"XGBoost RMSE: {xgb_rmse:.2f}, MAE: {xgb_mae:.2f}, R^2: {xgb_r2:.2f}")
"""

# RF regressor
numeric_transformer = Pipeline([
    # ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    # ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_feats),
    ('cat', categorical_transformer, categorical_feats)
])

rf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('rf', RandomForestRegressor(random_state=42))
])

param_grid_rf = {
    'rf__n_estimators': [50, 100, 150,200],
    'rf__max_depth': [None, 5, 10, 15, 20],
    'rf__min_samples_split': [2, 5, 8, 10, 12]
}

grid_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_rf.fit(X_train, y_train)
print("Best RF Params:", grid_rf.best_params_)
"""
y_pred_rf = grid_rf.predict(X_test)
import math
rf_rmse = math.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae  = mean_absolute_error(y_test, y_pred_rf)
rf_r2   = r2_score(y_test, y_pred_rf)
print(f"Random Forest RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}, R^2: {rf_r2:.2f}")

test['Item_Outlet_Sales'] = grid_rf.predict(test)
test.to_csv("pred_rf_F1.csv")
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

# Define base estimators using best models from GridSearchCV
base_estimators = [
    ('xgb', grid_xgb.best_estimator_),
    # ('cat', grid_cat.best_estimator_), # Environment issues - numpy vrsn compatibility with cat and xgb/rf
    ('rf',  grid_rf.best_estimator_)
]
# Meta-model
stacker = StackingRegressor(estimators=base_estimators, final_estimator=LinearRegression())
stacker.fit(X_train, y_train)

# Evaluate stacking model
y_pred_stack = stacker.predict(X_test)
stack_rmse = mean_squared_error(y_test, y_pred_stack)
stack_mae  = mean_absolute_error(y_test, y_pred_stack)
stack_r2   = r2_score(y_test, y_pred_stack)
print(f"Stacking RMSE: {stack_rmse:.2f}, MAE: {stack_mae:.2f}, R^2: {stack_r2:.2f}")

test['Item_Outlet_Sales'] = stacker.predict(test)
test.to_csv("pred_stacker_F1.csv")