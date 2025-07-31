import json
import pandas as pd
import joblib

# read config 
with open('config.json', 'r') as f:
    config = json.load(f)

DATA_PATH = config['cogs']['data_path']
MODEL_PATH = config['cogs']['model_path']
OUTPUT_PATH = config['cogs']['output_path']

COGS_MODEL = joblib.load(MODEL_PATH)

FEATURES = config['cogs']['predictors']

df = pd.read_csv(DATA_PATH)
df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df.set_index('date', inplace=True)

df['predicted_COGS'] = COGS_MODEL.predict(df[FEATURES])

df.to_csv(OUTPUT_PATH)