# Bank Marketing Prediction Lab Documentation

This documentation provides a step-by-step guide to a data science lab focused on predicting customer subscription to term deposits using Python. The lab covers data preprocessing, model training, model registration, and inference using MLflow and various machine learning libraries.

## Prerequisites

Before starting the lab, ensure that you have the following:

- Python environment set up with required libraries installed.
- Dataset: You will need the CSV file, `bank_marketing.csv`, containing bank marketing campaign data.

## Step 1: Importing Data

In this step, we load the bank marketing dataset using the Pandas library.
```python
import pandas as pd

# Load bank marketing data
data = pd.read_csv("data/bank_marketing.csv", sep=";")
```

## Step 2: Exploring Data

In this step, we'll explore the bank marketing dataset by examining the first few rows.

### Code:
```python
data.head()
data.info()
```

## Step 3: Data Preprocessing

In this step, we'll perform data preprocessing tasks to prepare the dataset for model training.

### Encoding Categorical Variables

We encode categorical variables and the target variable to convert them to numerical format.

### Code:
```python
from sklearn.preprocessing import LabelEncoder

# Create a copy for preprocessing
df = data.copy()

# Encode target variable (y)
df['y'] = df['y'].map({'no': 0, 'yes': 1})

# List of categorical columns to encode
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                    'contact', 'month', 'day_of_week', 'poutcome']

# Label encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
```

## Step 4: Data Visualization

In this step, we'll visualize the distribution of the target variable in the dataset.

### Code:
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=data, x='y')
plt.title('Subscription Distribution')
plt.show()
```

## Step 5: Define Target Variable

In this step, the target variable represents whether a customer subscribed to a term deposit.

### Code:
```python
# Target is already binary (0 = no, 1 = yes)
# Encoded in Step 3
```

### Explanation:
The target variable 'y' indicates whether a customer subscribed to a term deposit.
We use the map function to convert 'no' to 0 and 'yes' to 1.
Expected Output:
The df DataFrame will have a 'y' column with binary values (0 or 1).

## Step 6: Exploratory Data Analysis (EDA)

In this step, we'll perform Exploratory Data Analysis (EDA) by creating box plots to identify potential predictors of subscription.

### Code:
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

numerical_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate']

for idx, col in enumerate(numerical_features):
    row = idx // 3
    col_idx = idx % 3
    sns.boxplot(data=df, x='y', y=col, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'{col} vs Subscription')

plt.tight_layout()
plt.show()
```

## Step 7: Checking for Missing Data

In this step, we'll check for missing data within the dataset.

### Code:
```python
df.isna().any()
```

## Step 8: Data Splitting

In this step, we'll split the dataset into training, validation, and test sets to prepare for model training and evaluation.

### Code:
```python
from sklearn.model_selection import train_test_split

X = df.drop(['y'], axis=1)
y = df['y']

# Split out the training data
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=42)

# Split the remaining data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)
```

## Step 9: Building a Baseline Model

In this step, we'll create a baseline model using a random forest classifier and log its performance using MLflow.

### Code:
```python
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

# The predict method of sklearn's RandomForestClassifier returns a binary classification (0 or 1).
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to each class.

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:,1]

# mlflow.start_run creates a new MLflow run to track the performance of this model. 
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.

with mlflow.start_run(run_name='baseline_random_forest'):
    n_estimators = 100
    max_depth = 10
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
    predictions_test = model.predict_proba(X_test)[:,1]
    auc_score = roc_auc_score(y_test, predictions_test)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)
    
    # Use the area under the ROC curve as a metric.
    mlflow.log_metric('auc', auc_score)
    mlflow.log_metric('accuracy', accuracy)
    
    wrappedModel = SklearnModelWrapper(model)
    
    # Log the model with a signature that defines the schema of the model's inputs and outputs. 
    # When the model is deployed, this signature will be used to validate inputs.
    signature = infer_signature(X_train, wrappedModel.predict(None, X_train))

    # MLflow contains utilities to create a conda environment used to serve models.
    # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
    conda_env =  _mlflow_conda_env(
            additional_conda_deps=None,
            additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
            additional_conda_channels=None,
        )
    mlflow.pyfunc.log_model("random_forest_model",
                            python_model=wrappedModel,
                            conda_env=conda_env,
                            signature=signature)
```

### Explanation:
We create a random forest classifier model using scikit-learn's RandomForestClassifier.
The model is trained on the training data (X_train, y_train).
We log various information using MLflow, including model parameters (n_estimators, max_depth), the Area Under the ROC Curve (AUC) metric, and the model itself.
A wrapper class SklearnModelWrapper is used to make predictions using predict_proba, which returns class probabilities.
We also define a signature to validate inputs when the model is deployed.

Expected Output:
Model training details and metrics (e.g., AUC) will be logged in the MLflow run.
The trained model will be saved for future use.

## Step 10: Feature Importance Analysis

In this step, we analyze feature importance to identify which features have the most impact on predicting subscription.

### Code:
```python
feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)
```

### Explanation:
We calculate the feature importances using the trained random forest classifier model.
The model.feature_importances_ attribute provides the importance scores for each feature.
We create a DataFrame feature_importances to display the importances along with feature names.
Finally, we sort the DataFrame in descending order to identify the most important features.

Expected Output:
The output will be a table showing the feature importances in descending order, with the most important features at the top.

## Step 11: Model Registration in MLflow Model Registry

In this step, we'll register the trained model in the MLflow Model Registry for version tracking and management.

### Code:
```python
run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "baseline_random_forest"').iloc[0].run_id
model_name = "bank_marketing_subscription"
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)

# Registering the model takes a few seconds, so add a small delay
time.sleep(15)
```

### Explanation:
We retrieve the run ID of the MLflow run where the model was trained using mlflow.search_runs.
We specify the desired model name, in this case, "bank_marketing_subscription."
We use mlflow.register_model to register the model in the Model Registry. The path to the model is constructed using the run ID.
A delay is added to ensure the model registration process is completed.

Expected Output:
The trained model will be registered in the MLflow Model Registry under the specified model name ("bank_marketing_subscription").

### Note:
Model registration in the MLflow Model Registry allows for versioning and tracking of different model versions. It's a crucial step for managing and deploying machine learning models in a production environment.

## Step 12: Transitioning Model Version to Production

In this step, we'll transition the newly registered model version to the "Production" stage in the MLflow Model Registry.

### Code:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)
```

### Explanation:
We use the MlflowClient to interact with the MLflow Tracking Server programmatically.
The client.transition_model_version_stage method is used to transition the model version to the "Production" stage.

Expected Output:
The model version will be moved to the "Production" stage in the MLflow Model Registry.

### Note:
Transitioning a model version to "Production" indicates that it is ready for use in a production environment. You can now refer to the model using the path "models:/bank_marketing_subscription/production." This step is crucial for managing the deployment of machine learning models.

## Step 13: Model Inference and Evaluation

In this step, we'll load the production version of the model from the MLflow Model Registry and perform inference and evaluation.

### Code:
```python
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# Sanity-check: This should match the AUC logged by MLflow
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')
```

### Explanation:
We load the production version of the model from the MLflow Model Registry using mlflow.pyfunc.load_model.
We perform inference on the test data (X_test) using the loaded model.
We calculate and print the Area Under the ROC Curve (AUC) score to assess the model's performance.

Expected Output:
The AUC score will be printed, providing an evaluation of the model's performance on the test data.

### Note:
Loading the production model version allows us to make predictions on new data.
The AUC score is used here as an example metric for model evaluation. Depending on the problem, other evaluation metrics may be more appropriate.

## Step 14: Cleaning Up and Conclusion

In this final step, we'll wrap up the lab and perform any necessary clean-up tasks.

### Clean-Up Tasks:

- **Close Resources**: Ensure that any resources or connections used during the lab are properly closed or released.

- **Save Documentation**: Save this lab documentation for future reference or sharing with others.

### Conclusion:

In this lab, we've covered various aspects of the machine learning lifecycle, including data preparation, model training, evaluation, and model registry management. Here are the key takeaways:

- Data preparation is essential for training and evaluating machine learning models. Encoding categorical variables and splitting the data are crucial steps.

- Model training involves selecting an appropriate algorithm, training the model, and evaluating its performance using relevant metrics.

- Model deployment involves registering the model and transitioning it to the production stage for use in production environments.

- MLflow enables experiment tracking, model versioning, and reproducible ML workflows.
