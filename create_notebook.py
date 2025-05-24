import nbformat as nbf
import os

# Create a new notebook object
nb = nbf.v4.new_notebook()

# --- Markdown Cell: Title ---
markdown_cell_title = """\
# Health and Sleep Analysis"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown_cell_title))

# --- Code Cell: Imports ---
code_cell_imports = """\
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tabpfn import TabPFNClassifier # Corrected import name
import openml"""
nb.cells.append(nbf.v4.new_code_cell(code_cell_imports))

# --- Code Cell: Dataset Loading and Preprocessing Placeholder ---
code_cell_data_loading = """\
# Load the 'sleep' dataset (ID: 205) from OpenML as identified in the EDA phase
print("Loading dataset from OpenML...")
dataset = openml.datasets.get_dataset(205, download_data=True, download_qualities=True, download_features_meta_data=True)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='dataframe',
    target=dataset.default_target_attribute
)

# Combine X and y into a single DataFrame for convenience
df = X.copy()
if y is not None:
    df[dataset.default_target_attribute] = y

print("Dataset loaded successfully.")
print("First 5 rows of the dataset:")
print(df.head())
print("\\nShape of the dataset:", df.shape)
print("\\nTarget variable:", dataset.default_target_attribute)
print("\\nValue counts for the target variable:")
if y is not None:
    print(df[dataset.default_target_attribute].value_counts())
else:
    print("No default target variable 'y' was loaded separately.")

# Basic preprocessing: identify categorical and numerical features
if attribute_names:
    categorical_features = [attribute_names[i] for i, is_categorical in enumerate(categorical_indicator) if is_categorical and attribute_names[i] in df.columns]
    numerical_features = [attribute_names[i] for i, is_categorical in enumerate(categorical_indicator) if not is_categorical and attribute_names[i] in df.columns and attribute_names[i] != dataset.default_target_attribute]
else: 
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    if dataset.default_target_attribute in numerical_features:
        numerical_features.remove(dataset.default_target_attribute)
    if dataset.default_target_attribute in categorical_features:
        categorical_features.remove(dataset.default_target_attribute)

print(f"\\nIdentified categorical features: {categorical_features}")
print(f"Identified numerical features: {numerical_features}")

# Handle missing values (simple imputation for demonstration)
for col in numerical_features:
    if df[col].isnull().any():
        print(f"Imputing missing values in numerical feature '{col}' with mean.")
        df[col].fillna(df[col].mean(), inplace=True)

for col in categorical_features:
    if df[col].isnull().any():
        print(f"Imputing missing values in categorical feature '{col}' with mode.")
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\\nMissing values after imputation:")
print(df.isnull().sum())

if dataset.default_target_attribute and dataset.default_target_attribute in df.columns:
    y_processed = df[dataset.default_target_attribute]
    X_processed = df.drop(columns=[dataset.default_target_attribute])
    
    if dataset.default_target_attribute in numerical_features:
        numerical_features.remove(dataset.default_target_attribute)
    if dataset.default_target_attribute in categorical_features:
        categorical_features.remove(dataset.default_target_attribute)

    print(f"\\nShape of X_processed: {X_processed.shape}")
    print(f"Shape of y_processed: {y_processed.shape}")
else:
    print("\\nCould not define X_processed and y_processed as target attribute is missing or not found.")
    X_processed, y_processed = None, None
"""
nb.cells.append(nbf.v4.new_code_cell(code_cell_data_loading))

# --- Code Cell: Model Training and Evaluation Placeholder ---
code_cell_model_training = """\
# This cell will contain the model training and evaluation logic
# Ensure X_processed and y_processed are available
if 'X_processed' in locals() and 'y_processed' in locals() and X_processed is not None and y_processed is not None:
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed if y_processed.nunique() > 1 else None)

    numerical_pipeline = Pipeline([('scaler', StandardScaler())])
    categorical_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    current_numerical_features = [col for col in numerical_features if col in X_train.columns]
    current_categorical_features = [col for col in categorical_features if col in X_train.columns]
    
    print(f"\\nUsing numerical features for preprocessor: {current_numerical_features}")
    print(f"Using categorical features for preprocessor: {current_categorical_features}")

    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, current_numerical_features),
        ('categorical', categorical_pipeline, current_categorical_features)
    ], remainder='passthrough')

    print("\\n--- Training Logistic Regression Model ---")
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'))
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    
    print("\\nLogistic Regression - Test Set Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    class_labels = np.unique(y_train).astype(str) 
    print(classification_report(y_test, y_pred_lr, target_names=class_labels, zero_division=0))

    print("\\n--- Training TabPFN Model ---")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Shape of processed training data for TabPFN: {X_train_processed.shape}")
    if X_train_processed.shape[0] > 1000 or X_train_processed.shape[1] > 100 or y_train.nunique() > 10:
        print("Warning: Dataset size might exceed TabPFN default limits.")

    tabpfn_classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    tabpfn_classifier.fit(X_train_processed, y_train)
    y_pred_tabpfn = tabpfn_classifier.predict(X_test_processed)

    print("\\nTabPFN - Test Set Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_tabpfn):.4f}")
    print(classification_report(y_test, y_pred_tabpfn, target_names=class_labels, zero_division=0))
else:
    print("\\nSkipping model training as X_processed or y_processed are not available.")
"""
nb.cells.append(nbf.v4.new_code_cell(code_cell_model_training))

# Write the notebook to a file
notebook_filename = "health_sleep_analysis.ipynb"
with open(notebook_filename, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook '{notebook_filename}' created successfully.")

# Verify by listing files
print("\nFiles in current directory:")
for item in os.listdir("."):
    print(item)
# The problematic triple-quoted string was removed.
# Also corrected 'tabpfn.TabPFNClassifier' to 'TabPFNClassifier' in the import list within the notebook code.
# (Though it should be `from tabpfn import TabPFNClassifier`) - I've fixed this in the `code_cell_imports` string.
# Actually, looking at the TabPFN documentation, it is `from tabpfn import TabPFNClassifier`.
# The `code_cell_imports` string had `from tabpfn import TabPFNClassifier` which is correct.
# My comment was about `tabpfn.TabPFNClassifier` which was incorrect. The code in the cell is fine.

# The original script had a multiline comment at the very end that was not part of any Python string.
# That comment block seems to have been the source of the unterminated string literal error.
# I have removed any trailing comments that might have caused this.
# The script should now be valid Python.
