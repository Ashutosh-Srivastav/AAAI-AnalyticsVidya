# AAAI-AnalyticsVidya
Big Mart Sales Prediction
Problem Statement
Here: https://www.analyticsvidhya.com/datahack/contest/practice-problem-big-mart-sales-iii/

## Approach:
### 1. Data Exploration & Cleanup

The dataset includes key attributes like Item_Weight, Item_Visibility, Item_Type, Outlet_Size, Outlet_Location_Type, and Outlet_Type that influence sales trends.

Handling Missing Values:

Item_Weight: Missing values were imputed using the mean weight of the same Item_Identifier. If not available, the overall median weight was used.
Outlet_Size: Missing values were filled based on the most frequent Outlet_Size for each Outlet_Type.

Data Anomalies:

Item_Visibility: Several items had zero visibility, which was unrealistic. These were replaced with the mean visibility for the same Item_Identifier.
Item_Fat_Content: Inconsistent categories (e.g., LF, low fat, and Low Fat) were standardized to maintain uniformity.

Feature Creation:

Outlet_Age: Derived by subtracting Outlet_Establishment_Year from 2013.
Item_Identifier_Categories: Extracted first two characters of Item_Identifier to categorize items more effectively.

### 2. Feature Engineering & Selection

Encoding Categorical Variables: Item_Fat_Content, Outlet_Type, and Outlet_Location_Type, Item_Type and Item_Identifier_Categories.
Redundant columns (Item_Identifier, Outlet_Establishment_Year, Outlet_Identifier) were dropped based on exploratory analysis.


### 3. Model Selection & Training

As per EDA outcome -> No linear relationship to targer was observed and distribution of variables is skewed. Went for tree based regressors.
Tried linear and linear regularized models but negative sales values -> tried log trans, mapping to 0 in submissions, decided to try tree based models.

Random Forest Regressor
XGBoost Regressor: Applied gradient boosting to improve prediction accuracy.
Stacking Ensemble: Combined Random Forest and XGBoost with Linear Regression as the meta-model to enhance predictive performance.

Hyperparameter Tuning:
Used GridSearchCV to optimize model parameters such as n_estimators, max_depth, learning_rate, and min_child_weight.

Next steps:
Trying sequential deep neural nets.
Explore additional feature interactions using polynomial features.
