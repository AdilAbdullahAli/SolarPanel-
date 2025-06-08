import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import xgboost as xgb
source_train = 'C:\\Users\\adila\\OneDrive\\Desktop\\Project\\dataset\\train.csv'
source_test = 'C:\\Users\\adila\\OneDrive\\Desktop\\Project\\dataset\\test.csv' # Corrected path for test data
source_submission = 'C:\\Users\\adila\\OneDrive\\Desktop\\Project\\dataset\\submission1.csv'

# Load the dataframes correctly
train_df = pd.read_csv(source_train) # Load train.csv into train_df
test_df = pd.read_csv(source_test)   # Load test.csv into test_df
sample_submission_df = pd.read_csv(source_submission)

print(f"train_df loaded with shape: {train_df.shape}")
print(f"test_df loaded with shape: {test_df.shape}")
X = train_df.drop('efficiency', axis=1)
y = train_df['efficiency']
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
print(numerical_features)
numerical_features.remove('id')
categorical_features = X.select_dtypes(include='object').columns.tolist()
print(categorical_features)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough',force_int_remainder_cols=False
)
xgb_regressor = xgb.XGBRegressor(
    objective='reg:squarederror', # Objective for regression
    n_estimators=1000,           # Number of boosting rounds
    learning_rate=0.05,          # Step size shrinkage
    max_depth=6,                 # Maximum depth of a tree
    subsample=0.8,               # Subsample ratio of the training instance
    colsample_bytree=0.8,        # Subsample ratio of columns when constructing each tree
    random_state=42,
    n_jobs=-1                    # Use all available cores
)

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', xgb_regressor)])
print("\nPerforming K-Fold Cross-Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
custom_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1} ---")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    custom_score = 100 * (1 - rmse)

    rmse_scores.append(rmse)
    custom_scores.append(custom_score)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}, Custom Score: {custom_score:.2f}")

print(f"\nMean RMSE across folds: {np.mean(rmse_scores):.4f}")
print(f"Mean Custom Score across folds: {np.mean(custom_scores):.2f}")
X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred_val = model.predict(X_val)
print(y_pred_val,len(y_pred_val))
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
score_val = 100 * (1 - rmse_val)
print(f"Validation RMSE: {rmse_val}")
print(f"Validation Score: {score_val}")
test_predictions = model.predict(test_df)
print(test_df,len(test_df))
submission_df = pd.DataFrame({'id': test_df['id'], 'efficiency': test_predictions})
submission_df.to_csv(source_submission, index=False)
print("Submission file created successfully!")
print(submission_df)
#print(test_predictions)
