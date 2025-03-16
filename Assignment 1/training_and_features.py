from pymongo import MongoClient
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# MongoDB connection (Updated with your credentials)
client = MongoClient("mongodb+srv://divyagangotri03:iYMKfEmQftNCpo8e@cluster0.wqwmf.mongodb.net/?ssl=true")
db = client["hotel_guests"]
collection = db["dining_info"]

# Load Data
df = pd.DataFrame(list(collection.find()))

# Convert date columns
date_cols = ['check_in_date', 'check_out_date', 'order_time', 'prev_order_time']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Feature Engineering
df['order_month'] = df['order_time'].dt.month
df['check_in_day'] = df['check_in_date'].dt.dayofweek
df['check_out_day'] = df['check_out_date'].dt.dayofweek
df['stay_duration'] = (df['check_out_date'] - df['check_in_date']).dt.days

# Ensure order_time is available before calculating order_gap
if 'order_time' in df.columns:
    df['prev_order_time'] = df.groupby('customer_id')['order_time'].shift(1)
    df['order_gap'] = (df['order_time'] - df['prev_order_time']).dt.days.fillna(0)
else:
    print("⚠️ order_time column is missing in the dataset.")
    df['order_gap'] = 0

# Generate dish seasonality score
dish_seasonality = df.groupby(['dish', 'order_month'])['transaction_id'].count().reset_index()
dish_seasonality.columns = ['dish', 'order_month', 'dish_seasonality_score']
df = df.merge(dish_seasonality, on=['dish', 'order_month'], how='left')

# Splitting Data
train_df = df[(df['order_time'] >= '2024-01-01') & (df['order_time'] <= '2024-10-01')]
test_df = df[df['order_time'] > '2024-10-01']

# Customer-level features
customer_features = df.groupby('customer_id').agg(
    total_orders_per_customer=('transaction_id', 'count'),
    avg_spend_per_customer=('price_for_1', 'mean'),
    total_qty_per_customer=('Qty', 'sum'),
    avg_stay_per_customer=('stay_duration', 'mean'),
    avg_order_gap=('order_gap', 'mean')
).reset_index()
customer_features.to_excel('customer_features.xlsx', index=False)

# Cuisine-Level Features
cuisine_features = df.groupby('Preferred Cusine').agg(
    avg_price_per_cuisine=('price_for_1', 'mean'),
    total_orders_per_cuisine=('transaction_id', 'count')
).reset_index()
cuisine_features.to_excel('cuisine_features.xlsx', index=False)

# Drop unnecessary columns before training
drop_cols = ['_id', 'check_in_date', 'check_out_date', 'order_time', 'prev_order_time']
train_df = train_df.drop(columns=[col for col in drop_cols if col in train_df.columns], errors='ignore')
test_df = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns], errors='ignore')

# Encoding categorical features
categorical_cols = ['Preferred Cusine']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_train = encoder.fit_transform(train_df[categorical_cols])
encoded_test = encoder.transform(test_df[categorical_cols])
train_df = pd.concat([train_df.drop(columns=categorical_cols), pd.DataFrame(encoded_train)], axis=1)
test_df = pd.concat([test_df.drop(columns=categorical_cols), pd.DataFrame(encoded_test)], axis=1)
joblib.dump(encoder, 'encoder.pkl')

# Label Encoding Target
label_encoder = LabelEncoder()
train_df['dish'] = label_encoder.fit_transform(train_df['dish'])
test_df['dish'] = label_encoder.transform(test_df['dish'])
joblib.dump(label_encoder, 'label_encoder.pkl')

# Train Model
X_train, y_train = train_df.drop(columns=['dish']), train_df['dish']
model = xgb.XGBClassifier(
    objective="multi:softmax",
    eval_metric="mlogloss",
    learning_rate=0.01,
    max_depth=2,
    n_estimators=15,
    subsample=0.5,
    colsample_bytree=0.2,
    reg_lambda=72,
    reg_alpha=30,
    random_state=42,
    enable_categorical=True  # Fix XGBoost categorical error
)
model.fit(X_train, y_train)
joblib.dump(model, 'xgb_model.pkl')
