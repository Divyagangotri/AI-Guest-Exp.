from pymongo import MongoClient
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt

# MongoDB connection
client = MongoClient("mongodb+srv://divyagangotri03:iYMKfEmQftNCpo8e@cluster0.wqwmf.mongodb.net/?ssl=true")

# Database and collection
db = client["hotel_guests"]
collection = db["dining_info"]

# Load data from MongoDB into DataFrame
df_from_mongo = pd.DataFrame(list(collection.find()))

# Work with a copy of the DataFrame
df = df_from_mongo.copy()

# Convert columns to datetime format with error handling
df['check_in_date'] = pd.to_datetime(df['check_in_date'], errors='coerce')
df['check_out_date'] = pd.to_datetime(df['check_out_date'], errors='coerce')
df['order_time'] = pd.to_datetime(df['order_time'], errors='coerce')

# Extract month, day of the week, and calculate stay duration
 
df['check_in_day'] = df['check_in_date'].dt.dayofweek
df['check_out_day'] = df['check_out_date'].dt.dayofweek
df['check_in_month'] = df['check_in_date'].dt.month
df['check_out_month'] = df['check_out_date'].dt.month
df['stay_duration'] = (df['check_out_date'] - df['check_in_date']).dt.days  # Added feature

# Derive features based on historical data
features_df = df[df['order_time'] < '2024-01-01'].copy()

train_df = df[(df['order_time'] >= '2024-01-01') & (df['order_time'] <= '2024-10-01')]
test_df = df[df['order_time'] > '2024-10-01']  # Pseudo prediction dataset

# Customer-level features
customer_features = features_df.groupby('customer_id').agg(
    total_orders_per_customer=('transaction_id', 'count'),
    avg_spend_per_customer=('price_for_1', 'mean')
).reset_index()
customer_features.to_excel('customer_features.xlsx', index=False)

# Most frequent dish per customer
customer_dish = features_df.groupby('customer_id')['dish'].agg(lambda x: x.mode()[0]).reset_index()
customer_dish.rename(columns={'dish': 'customer_dish'}, inplace=True)
customer_dish.to_excel('customer_dish.xlsx', index=False)

 

# Cuisine-level features
cuisine_features = features_df.groupby('Preferred Cusine').agg(
    total_orders_per_cuisine=('transaction_id', 'count')
).reset_index()
cuisine_features.to_excel('cuisine_features.xlsx', index=False)


# Most popular dish per cuisine
cuisine_popular_dish = features_df.groupby('Preferred Cusine')['dish'].agg(lambda x: x.mode()[0]).reset_index()
cuisine_popular_dish.rename(columns={'dish': 'cuisine_popular_dish'}, inplace=True)

cuisine_popular_dish.to_excel('cuisine_popular_dish.xlsx', index=False)

 
# Merging features to train_df
train_df = train_df.merge(customer_features, on='customer_id', how='left')
train_df = train_df.merge(customer_dish, on='customer_id', how='left')
train_df = train_df.merge(cuisine_features, on='Preferred Cusine', how='left')
train_df = train_df.merge(cuisine_popular_dish, on='Preferred Cusine', how='left')
 

# Added stay_duration to train set
train_df['stay_duration'] = df['stay_duration']

# Drop unnecessary columns
train_df.drop(['_id','transaction_id','customer_id','price_for_1', 'Qty','order_time','check_in_date','check_out_date'], axis=1, inplace=True)



# One-hot encode categorical features

for df in [train_df, test_df]:
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        df[col] = df[col].astype('category')


categorical_cols = ['Preferred Cusine', 'customer_dish', 'cuisine_popular_dish']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_array = encoder.fit_transform(train_df[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))
train_df = pd.concat([train_df.drop(columns=categorical_cols), encoded_df], axis=1)

# Prepare the test set with similar transformations
test_df = test_df.merge(customer_features, on='customer_id', how='left')
test_df = test_df.merge(customer_dish, on='customer_id', how='left')
test_df = test_df.merge(cuisine_features, on='Preferred Cusine', how='left')
test_df = test_df.merge(cuisine_popular_dish, on='Preferred Cusine', how='left')
 

# Added stay_duration to test set
test_df['stay_duration'] = df['stay_duration']

test_df.drop(['_id','transaction_id','customer_id','price_for_1','Qty','order_time','check_in_date','check_out_date'], axis=1, inplace=True)

encoded_test = encoder.transform(test_df[categorical_cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))
test_df = pd.concat([test_df.drop(columns=categorical_cols), encoded_test_df], axis=1)

joblib.dump(encoder, 'encoder.pkl')
# Drop NaN rows
train_df = train_df.dropna(subset=['dish'])
test_df = test_df.dropna(subset=['dish'])

# Label encode the target column 'dish'
label_encoder = LabelEncoder()
train_df['dish'] = label_encoder.fit_transform(train_df['dish'])
test_df['dish'] = label_encoder.transform(test_df['dish'])

joblib.dump(label_encoder, 'label_encoder.pkl')
# Split into features (X) and target (y)
X_train = train_df.drop(columns=['dish'])
y_train = train_df['dish']
X_test = test_df.drop(columns=['dish'])
y_test = test_df['dish']

# Train the XGBoost model with lower accuracy
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    eval_metric="mlogloss",
    learning_rate=0.01,  # Increased learning rate
    max_depth=2,  # Reduced from 10
    n_estimators=15,  # Reduced from 300
    subsample=0.5,  # Reduced from 0.9
    colsample_bytree=0.2,  # Reduced from 0.8
    reg_lambda=72,  # Added L2 regularization
    reg_alpha=30,  # Added L1 regularization
    random_state=42
)

xgb_model.fit(X_train, y_train)

joblib.dump(xgb_model, 'xgb_model.pkl')
pd.DataFrame(X_train.columns).to_excel('features.xlsx')
# Make predictions
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy = ', accuracy)

# Compute log loss
y_pred_prob = xgb_model.predict_proba(X_test)
logloss = log_loss(y_test, y_pred_prob)
print('logloss = ', logloss)

# Plot feature importance
xgb.plot_importance(xgb_model, max_num_features=5)
plt.show()
