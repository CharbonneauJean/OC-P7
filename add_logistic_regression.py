import nbformat as nbf
import os

notebook_filename = "health_sleep_analysis.ipynb"

# Read the existing notebook
print(f"Reading existing notebook: {notebook_filename}")
try:
    with open(notebook_filename, 'r') as f:
        nb = nbf.read(f, as_version=4)
except FileNotFoundError:
    print(f"Notebook '{notebook_filename}' not found. This script expects the notebook from previous steps.")
    exit(1)

# Find the index of the last data preprocessing cell
# This is typically the cell with the source containing "X_test_processed = preprocessor.transform(X_test)"
# or the markdown cell "## 5. Apply Preprocessing"
preprocessing_cell_index = -1
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "X_test_processed = preprocessor.transform(X_test)" in cell.source:
        preprocessing_cell_index = i
        break
    # Fallback to markdown cell if code cell structure changes slightly
    elif cell.cell_type == 'markdown' and "## 5. Apply Preprocessing" in cell.source:
         # If this is a markdown cell, the actual code is likely in the next cell
        if i + 1 < len(nb.cells) and nb.cells[i+1].cell_type == 'code':
            preprocessing_cell_index = i + 1
        else:
            preprocessing_cell_index = i # Or just use the markdown cell's index


if preprocessing_cell_index == -1:
    print("Warning: Could not find the last data preprocessing cell. Appending new cells at the end.")
    # This might happen if the content of the last preprocessing cell changed.
    # As a robust fallback, find the last cell that defines X_train_processed or X_test_processed
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and ("X_train_processed =" in cell.source or "X_test_processed =" in cell.source):
            preprocessing_cell_index = i # Take the last occurrence
    if preprocessing_cell_index == -1:
        preprocessing_cell_index = len(nb.cells) -1 # Append at the very end if still not found


# New cells to be inserted for Logistic Regression
new_lr_cells = []

# --- Markdown Cell: Logistic Regression Model ---
markdown_lr_intro = """\
## 6. Traditional Model: Logistic Regression

We will now train a traditional classification model, Logistic Regression, using the preprocessed training data. This will serve as a baseline model."""
new_lr_cells.append(nbf.v4.new_markdown_cell(markdown_lr_intro))

# --- Code Cell: Train Logistic Regression Model ---
code_lr_train = """\
# Ensure X_train_processed and y_train are available from the previous preprocessing steps
if 'X_train_processed' in locals() and 'y_train' in locals() and X_train_processed is not None and y_train is not None:
    print("Training Logistic Regression model...")
    
    # Initialize Logistic Regression model
    # Increased max_iter for convergence, especially with scaled data.
    # Using 'solver' explicitly can also be good practice e.g. 'liblinear' for smaller datasets or 'lbfgs'
    lr_model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear') 
    
    # Train the model
    # TabPFN expects numpy array, ensure X_train_processed is suitable
    # If X_train_processed_df was created, use X_train_processed (the numpy array) for scikit-learn consistency
    # or ensure that X_train_processed_df is what you intend to use (if it exists and is preferred)
    
    # The variable X_train_processed should be a NumPy array if preprocessor.fit_transform was used directly.
    # If X_train_processed_df was created and preferred, that variable should be used.
    # Assuming X_train_processed is the NumPy array from preprocessor.
    
    lr_model.fit(X_train_processed, y_train)
    
    print("Logistic Regression model trained successfully.")
    print("Model details:", lr_model)
else:
    print("Skipping Logistic Regression model training as X_train_processed or y_train are not available.")
    lr_model = None # Ensure lr_model exists even if training is skipped
"""
new_lr_cells.append(nbf.v4.new_code_cell(code_lr_train))

# Insert new cells after the preprocessing cell
insert_position = preprocessing_cell_index + 1
nb.cells = nb.cells[:insert_position] + new_lr_cells + nb.cells[insert_position:]

print(f"Inserted {len(new_lr_cells)} new cells for Logistic Regression training after cell index {preprocessing_cell_index}.")

# Write the updated notebook to a file
with open(notebook_filename, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook '{notebook_filename}' updated successfully with Logistic Regression training steps.")

# Verify by listing files
print("\nFiles in current directory:")
for item in os.listdir("."):
    print(item)
