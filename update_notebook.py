import nbformat as nbf
import os

notebook_filename = "health_sleep_analysis.ipynb"

# Read the existing notebook
print(f"Reading existing notebook: {notebook_filename}")
try:
    with open(notebook_filename, 'r') as f:
        nb = nbf.read(f, as_version=4)
except FileNotFoundError:
    print(f"Notebook '{notebook_filename}' not found. Creating a new one.")
    nb = nbf.v4.new_notebook()
    # Add initial cells if it's a new notebook (title, imports)
    markdown_cell_title = "# Health and Sleep Analysis"
    nb.cells.append(nbf.v4.new_markdown_cell(markdown_cell_title))
    code_cell_imports = """\
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tabpfn import TabPFNClassifier
import openml
from sklearn.impute import SimpleImputer""" # Added SimpleImputer
    nb.cells.append(nbf.v4.new_code_cell(code_cell_imports))


# Find the index of the import cell
import_cell_index = -1
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        if 'import openml' in cell.source and 'TabPFNClassifier' in cell.source :
            import_cell_index = i
            if 'from sklearn.impute import SimpleImputer' not in cell.source:
                cell.source = "from sklearn.impute import SimpleImputer\n" + cell.source
            break

if import_cell_index == -1:
    print("Warning: Could not find the main import cell. Appending new cells at the end.")
    import_cell_index = len(nb.cells) -1 

new_cells = []

markdown_data_loading = """\
## 1. Load the Dataset

We will load the 'sleep' dataset (ID: 205) from OpenML. This dataset was identified during the EDA phase. We will also separate features (X) and the target variable (y), which is 'danger_index'."""
new_cells.append(nbf.v4.new_markdown_cell(markdown_data_loading))

code_data_loading = """\
print("Loading 'sleep' dataset (ID: 205) from OpenML...")
dataset = openml.datasets.get_dataset(205, download_data=True, download_qualities=True, download_features_meta_data=True)

target_column = 'danger_index' 

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='dataframe',
    target=target_column 
)

df = X.copy()
if y is not None:
    df[target_column] = y

print("Dataset loaded successfully.")
print("First 5 rows of the combined DataFrame (for inspection):")
print(df.head())
print("\\nShape of features X:", X.shape)
print("Shape of target y:", y.shape)
print("\\nTarget variable column name:", target_column)
print("\\nValue counts for the target variable:")
if y is not None:
    print(y.value_counts())
else:
    print("Target variable 'y' could not be loaded.")

categorical_features = [attribute_names[i] for i, is_cat in enumerate(categorical_indicator) 
                        if is_cat and attribute_names[i] in X.columns]
numerical_features = [attribute_names[i] for i, is_cat in enumerate(categorical_indicator) 
                      if not is_cat and attribute_names[i] in X.columns]

print(f"\\nIdentified categorical features in X: {categorical_features}")
print(f"Identified numerical features in X: {numerical_features}")

print("\\nMissing values in X before imputation:")
print(X.isnull().sum()[X.isnull().sum() > 0])
print("\\nMissing values in y before imputation (if any):")
if y is not None:
    print(y.isnull().sum())
else:
    print("y is None")"""
new_cells.append(nbf.v4.new_code_cell(code_data_loading))

markdown_missing_values = """\
## 2. Handle Missing Values

We will impute missing values using `SimpleImputer` from scikit-learn.
- For **numerical features**, we'll use the 'median' strategy.
- For **categorical features**, we'll use the 'most_frequent' strategy."""
new_cells.append(nbf.v4.new_markdown_cell(markdown_missing_values))

code_missing_values = """\
numerical_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

X_imputed = X.copy()

if numerical_features:
    print("\\nImputing numerical features...")
    X_imputed[numerical_features] = numerical_imputer.fit_transform(X[numerical_features])
else:
    print("\\nNo numerical features to impute.")

if categorical_features:
    print("Imputing categorical features...")
    X_imputed[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
else:
    print("No categorical features to impute.")

print("\\nMissing values in X after imputation:")
print(X_imputed.isnull().sum()[X_imputed.isnull().sum() > 0])

if y is not None and y.isnull().any():
    print(f"\\nTarget variable '{target_column}' has {y.isnull().sum()} missing values.")
else:
    print(f"\\nTarget variable '{target_column}' has no missing values or y is None.")
"""
new_cells.append(nbf.v4.new_code_cell(code_missing_values))

markdown_feature_pipeline = """\
## 3. Feature Engineering Pipeline

We'll create a `ColumnTransformer` to apply different preprocessing steps to numerical and categorical features.
- **Numerical features**: Will be scaled using `StandardScaler` (after median imputation).
- **Categorical features**: Will be encoded using `OneHotEncoder` (after most_frequent imputation). `handle_unknown='ignore'` is used to prevent errors during transform if test data has new categories."""
new_cells.append(nbf.v4.new_markdown_cell(markdown_feature_pipeline))

code_feature_pipeline = """\
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
    remainder='passthrough' 
)

print("ColumnTransformer created successfully.")
print("Numerical features for transformer:", numerical_features)
print("Categorical features for transformer:", categorical_features)
"""
new_cells.append(nbf.v4.new_code_cell(code_feature_pipeline))

markdown_split_data = """\
## 4. Split the Data

The dataset (features `X_imputed` and target `y`) will be split into training and testing sets. We'll use a 80/20 split and set a `random_state` for reproducibility."""
new_cells.append(nbf.v4.new_markdown_cell(markdown_split_data))

code_split_data = """\
if y is not None:
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y if y.nunique() > 1 else None 
    )
    print("Data split into training and testing sets.")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
else:
    print("Cannot split data as target variable y is not available.")
    X_train, X_test, y_train, y_test = None, None, None, None
"""
new_cells.append(nbf.v4.new_code_cell(code_split_data))

markdown_apply_preprocessing = """\
## 5. Apply Preprocessing

The `ColumnTransformer` (preprocessor) will be fitted on the training data (`X_train`) only, to prevent data leakage from the test set. Then, both `X_train` and `X_test` will be transformed."""
new_cells.append(nbf.v4.new_markdown_cell(markdown_apply_preprocessing))

code_apply_preprocessing = """\
if X_train is not None:
    print("Fitting preprocessor on X_train and transforming X_train...")
    X_train_processed = preprocessor.fit_transform(X_train)
    print("Transforming X_test...")
    X_test_processed = preprocessor.transform(X_test)

    try:
        feature_names_out = preprocessor.get_feature_names_out()
        X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names_out, index=X_train.index)
        X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_out, index=X_test.index)
        
        print("\\nShape of X_train_processed:", X_train_processed_df.shape)
        print("First 5 rows of X_train_processed_df:")
        print(X_train_processed_df.head())
        
        print("\\nShape of X_test_processed:", X_test_processed_df.shape)
    except Exception as e:
        print(f"Could not get feature names out or convert to DataFrame: {e}")
        print("X_train_processed and X_test_processed are likely NumPy arrays.")
        print("\\nShape of X_train_processed (array):", X_train_processed.shape)
        print("Shape of X_test_processed (array):", X_test_processed.shape)
else:
    print("Skipping preprocessing application as X_train is not available.")
    X_train_processed, X_test_processed = None, None

"""
new_cells.append(nbf.v4.new_code_cell(code_apply_preprocessing))

# Logic to insert new cells and remove old placeholders
original_cells = nb.cells
insert_position = import_cell_index + 1

# Filter out the old placeholder cells based on their specific content
final_cells = []
placeholder_1_content_start = "# Load the 'sleep' dataset (ID: 205) from OpenML as identified in the EDA phase"
placeholder_2_content_start = "# This cell will contain the model training and evaluation logic"
placeholders_removed_count = 0

# It's safer to build a new list of cells, excluding the old placeholders,
# and then inserting the new cells in the correct position relative to the import cell.

# First, identify the cells to keep, excluding the old placeholders
cells_to_keep_initially = []
for cell in original_cells:
    is_placeholder_to_remove = False
    if cell.cell_type == 'code':
        if placeholder_1_content_start in cell.source:
            is_placeholder_to_remove = True
            print(f"Marking old data loading placeholder cell for removal.")
        elif placeholder_2_content_start in cell.source:
            is_placeholder_to_remove = True
            print(f"Marking old model training placeholder cell for removal.")
    
    if not is_placeholder_to_remove:
        cells_to_keep_initially.append(cell)
    else:
        placeholders_removed_count += 1

nb.cells = cells_to_keep_initially
print(f"Removed {placeholders_removed_count} old placeholder cell(s) based on content.")

# Now, find the import cell index again in the potentially modified list
recalculated_import_cell_index = -1
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'import openml' in cell.source and 'TabPFNClassifier' in cell.source:
        recalculated_import_cell_index = i
        break

if recalculated_import_cell_index != -1:
    insert_position = recalculated_import_cell_index + 1
    nb.cells = nb.cells[:insert_position] + new_cells + nb.cells[insert_position:]
    print(f"Inserted {len(new_cells)} new cells after the import cell (new index {recalculated_import_cell_index}).")
else:
    # This case should ideally not happen if the import cell was present and not removed.
    print("Error: Could not re-find import cell after removing placeholders. Appending new cells to the end of current structure.")
    nb.cells.extend(new_cells)


with open(notebook_filename, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook '{notebook_filename}' updated successfully.")

print("\nFiles in current directory:")
for item in os.listdir("."):
    print(item)
