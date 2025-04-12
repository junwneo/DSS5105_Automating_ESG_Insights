
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


#  Loading storage data
def load_data():
    df=pd.read_csv("data1.csv")
    # Renaming columns
    column_mapping = {col: col.strip().lower().replace('&', 'and') for col in df.columns}
    return df.rename(columns=column_mapping)

# Advanced preprocessing
def advanced_preprocessing(df):
    df = df[df['esg_score'].notna()].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Feature engineering of different year
    df['years_active'] = df.groupby('company')['year'].transform(lambda x: x - x.min() + 1)
    df['year_diff'] = df.groupby('company')['year'].transform(lambda x: x.diff().fillna(1))
    
    # Auto fill missing values
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        df[col] = df.groupby('company')[col].transform(
            lambda x: x.fillna(x.mean() if x.mean() > 0 else 0)
        )
    
    return df

# Define the model
models = {
         'XGBoost1': make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    XGBRegressor(
        n_estimators=1000,
        max_depth=2,
        learning_rate=0.1,
        random_state=42
    )),
}

# preapre data
df = load_data()
processed_df = advanced_preprocessing(df)

# features engineering
X = processed_df.drop(columns=['esg_score', 'company', 'year'])
y = processed_df['esg_score']
groups = processed_df['company']

# Final model training 
final_model = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    XGBRegressor(
        n_estimators=1000,
        max_depth=2,
        learning_rate=0.1,
        random_state=42
    )
).fit(X, y)




# Prediction model
def predict_esg(model, current_data):
    # generate time feather (not keep company/year）
    current_data = current_data.copy()
    current_data['years_active'] = current_data['year'] - current_data['year'].min() + 1
    current_data['year_diff'] = 1
    
    required_features = X.columns.tolist()  
    missing_features = set(required_features) - set(current_data.columns)
    
    # Fill in missing features
    for feat in missing_features:
        current_data[feat] = 0
    
    return model.predict(current_data[required_features])

# Prediction model
def predict_esg(model, current_data):
    # generate time feather (not keep company/year）
    current_data = current_data.copy()
    current_data['years_active'] = current_data['year'] - current_data['year'].min() + 1
    current_data['year_diff'] = 1
    
    required_features = X.columns.tolist()  
    missing_features = set(required_features) - set(current_data.columns)
    
    # Fill in missing features
    for feat in missing_features:
        current_data[feat] = 0
    
    return model.predict(current_data[required_features])

def standardize_columns(df):
    column_mapping = {col: col.strip().lower().replace('&', 'and') for col in df.columns}
    return df.rename(columns=column_mapping)


# Process latest data 
def load_latest_row():
    # read csv
    df = pd.read_csv("consolidated_esg_single_row.csv")
    df = standardize_columns(df)
    latest_row = df.iloc[-1].to_dict()
    
    # trun in DataFrame
    latest_df = pd.DataFrame([latest_row])
    
    # add year column if not exist
    if 'year' not in latest_df.columns:
        latest_df['year'] = 2024  # 添加默认年份
    
    return latest_df

# load ESG score
def load_esg_scores():
    esg_df = pd.read_csv("esg_scored_result.csv")
    last_row = esg_df.iloc[-1]
    esg_score_1 = last_row.iloc[-2]  
    esg_score_2 = last_row.iloc[-1]  
    return esg_score_1, esg_score_2

# process new data 
latest_data = load_latest_row()

# get ESG scores
esg_score_1, esg_score_2 = load_esg_scores()

#add nessasary columns
latest_data['esg_score'] = esg_score_1
latest_data['grade'] = esg_score_2

processed_latest = advanced_preprocessing(latest_data)

pred_score = predict_esg(final_model, processed_latest)
print(f"Predict value ：{pred_score[0]:.2f}")

def convert_grade(score):
    if pd.isna(score):
        return None
    if score < 30:
        return 'D'
    elif score < 40:
        return 'C-'
    elif score < 50:
        return 'C'
    elif score < 60:
        return 'C+'
    elif score < 67:
        return 'B-'
    elif score < 73:
        return 'B'
    elif score < 78:
        return 'B+'
    elif score < 83:
        return 'A-'
    elif score < 88:
        return 'A'
    else:
        return 'A+'
    

    # Convert the predicted score to a letter grade
predicted_grade = convert_grade(pred_score[0])
print(f"Predicted grade: {predicted_grade}")
