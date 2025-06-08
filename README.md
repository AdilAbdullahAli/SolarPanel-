Solar Panel Efficiency Prediction

---

 Project Overview

This project aims to predict the efficiency of solar panels using a machine learning model. By analyzing various operational and environmental parameters, the model identifies patterns that influence panel performance and helps in predicting potential degradation or failure. This can be crucial for proactive maintenance and optimizing energy generation in solar farms.

The problem is framed as a regression task, where the goal is to predict a continuous numerical value (efficiency).

 Dataset

The project utilizes two primary datasets:
 `train.csv`: Contains operational parameters, environmental conditions, and the target `efficiency` for training the model.
 `test.csv`: Contains similar parameters (excluding `efficiency`) for which predictions need to be made.
 `sample_submission.csv`: A template for the submission file.

Key Features include (but are not limited to):
 `string_id`, `panel_id`: Identifiers for solar panel strings and individual panels.
 `temperature`, `humidity`, `irradiance`: Environmental conditions.
 `voltage`, `current`, `module_temperature`: Electrical and module-specific parameters.
 `panel_age`, `maintenance_count`, `soiling_ratio`: Panel-specific characteristics.
 `cloud_coverage`, `wind_speed`, `pressure`: Additional environmental factors.
 `error_code`: Status codes indicating operational issues.
 `installation_type`: Type of solar panel installation (e.g., fixed, tracking).
 `efficiency`: The target variable, representing the panel's performance (in the training set).

## Solution Approach

The solution employs a machine learning pipeline built with `scikit-learn` and `XGBoost` to ensure robust preprocessing, model training, and prediction.

### 1. Data Loading and Initial Preparation

 `train.csv` and `test.csv` are loaded into pandas DataFrames.
 The `efficiency` column is separated as the target variable (`y`), while the remaining columns form the features (`X`).
 The `id` columns from both `X` (training features) and `test_df` are dropped as they are identifiers and hold no predictive power. The `id`s from `test_df` are stored separately for generating the submission file.

### 2. Preprocessing Pipeline

A comprehensive preprocessing pipeline handles various data challenges:

Missing Value Imputation:
  Numerical Features:** Missing values in numerical columns are filled using the **median** strategy. The median is robust to outliers and is a good choice for sensor data.
    Categorical Features:** Missing values in categorical columns are imputed with a constant string `'missing'`, treating them as a distinct category.
Feature Scaling:
     Numerical Features:All numerical features are scaled using `StandardScaler`. This transforms the data to have a mean of 0 and a standard deviation of 1. Scaling is crucial for many machine learning algorithms (like SVMs, K-Nearest Neighbors, and even tree-based models indirectly benefit) as it prevents features with larger scales from dominating the learning process.
*Categorical Encoding:
    *Categorical Features: Categorical features are converted into a numerical format using `OneHotEncoder`. This creates a new binary column for each unique category, allowing the model to understand these non-numeric inputs. `handle_unknown='ignore'` is used to gracefully handle categories that might appear in the test set but not in the training set.
`ColumnTransformer`**: This powerful `scikit-learn` tool is used to apply different preprocessing steps to different columns (numerical and categorical) simultaneously within a unified pipeline.

3. Machine Learning Model: XGBoost Regressor

The core of the prediction model is an XGBoost Regressor (`xgboost.XGBRegressor`). XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It's known for its speed and performance in structured/tabular data problems.

The `XGBRegressor` is configured with:
 `objective='reg:squarederror'`: Specifies the regression objective function.
 `n_estimators=1000`: The number of boosting rounds (trees).
 `learning_rate=0.05`: Shrinkage factor applied to each tree's contribution.
 `max_depth=6`: Maximum depth of a tree.
 `subsample=0.8`: Fraction of samples used for fitting the trees.
 `colsample_bytree=0.8`: Fraction of features used for fitting the trees.
 `random_state=42`: For reproducibility of results.
 `n_jobs=-1`: Utilizes all available CPU cores for parallel processing.
The `XGBRegressor` is integrated into a `Pipeline` directly after the `ColumnTransformer`, ensuring that all preprocessing steps are applied automatically before the model sees the data.

4. Cross-Validation

To robustly evaluate the model's performance and ensure it generalizes well to unseen data, **K-Fold Cross-Validation** (`KFold` with 5 splits) is performed.
* The training data is split into 5 folds.
* In each fold, the model is trained on 4 folds and validated on the remaining 1 fold.
* The performance metrics (RMSE and Custom Score) are calculated for each fold, and the mean scores across all folds provide a reliable estimate of the model's performance.

Evaluation Metrics:
RMSE (Root Mean Squared Error): A common metric for regression, indicating the average magnitude of the errors. Lower RMSE is better.
* Custom Score (100 * (1 - RMSE)): A problem-specific score where higher values indicate better performance.

 5. Final Model Training and Prediction

After cross-validation, the model is trained on the **entire available training dataset** (`X` and `y`). This allows the model to learn from all 20,000 data points, maximizing its knowledge base before making predictions on the completely unseen test set.

Finally, predictions are generated for the `test_df_for_prediction` (which is the preprocessed `test.csv` data). The predictions are then clipped to be within the valid efficiency range of 0 to 100, ensuring realistic outputs.

 6. Submission File Generation

The predicted efficiency values are combined with the original `id`s from the `test.csv` to create a `submission.csv` file in the required format (`id`, `efficiency`), ready for submission.

 How to Run the Code

1.  Save the Code: Save the provided Python script (e.g., as `predict_efficiency.py`).
2.  Place Data Files: Ensure `train.csv`, `test.csv`, and `sample_submission.csv` are located in the specified directory: `C:\Users\adila\OneDrive\Desktop\Project\dataset\`. If your data is elsewhere, update the `source_train`, `source_test`, and `source_submission` variables in the script accordingly.
3.  Install Dependencies: Make sure you have all the necessary Python libraries installed. You can install them using pip:
    bash
    pip install pandas numpy scikit-learn xgboost
4.  Run the Script: Execute the Python script from your terminal:
    python predict_efficiency.py


The script will print the shapes of the loaded data, feature lists, cross-validation results, and finally generate a `submission1.csv` file.
