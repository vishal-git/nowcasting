import json
import pandas as pd
import joblib

# read config 
with open('config.json', 'r') as f:
    config = json.load(f)

DATA_PATH = config['marketing_cost']['data_path']
MODEL_PATH = config['marketing_cost']['model_path']
OUTPUT_PATH = config['marketing_cost']['output_path']
FEATURES = config['marketing_cost']['predictors']

MARKETING_COST_MODEL = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df.set_index('date', inplace=True)

df['predicted_marketing_cost'] = MARKETING_COST_MODEL.predict(df[FEATURES])

df['predicted_marketing_cost'].to_csv(OUTPUT_PATH, index=True)