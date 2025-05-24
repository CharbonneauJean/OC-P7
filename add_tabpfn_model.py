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

# Find the index of the last Logistic Regression model cell
# This is typically the cell with the source containing "lr_model.fit(X_train_processed, y_train)"
# or the markdown cell "## 6. Traditional Model: Logistic Regression"
lr_model_cell_index = -1
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "lr_model.fit(X_train_processed, y_train)" in cell.source:
        lr_model_cell_index = i
        break
    # Fallback to markdown cell if code cell structure changes slightly
    elif cell.cell_type == 'markdown' and "## 6. Traditional Model: Logistic Regression" in cell.source:
        # If this is a markdown cell, the actual code is likely in the next cell
        if i + 1 < len(nb.cells) and nb.cells[i+1].cell_type == 'code':
            lr_model_cell_index = i + 1
        else:
            lr_model_cell_index = i # Or just use the markdown cell's index


if lr_model_cell_index == -1:
    print("Warning: Could not find the Logistic Regression model training cell. Appending new cells at the end.")
    lr_model_cell_index = len(nb.cells) -1 # Append at the very end if still not found


# New cells to be inserted for TabPFN Model
new_tabpfn_cells = []

# --- Markdown Cell: TabPFN Model ---
markdown_tabpfn_intro = """\
## 7. TabPFN Model

Next, we will train a `TabPFNClassifier`. TabPFN is a pre-trained model that can achieve good performance on small tabular datasets without extensive hyperparameter tuning. It's designed to be efficient for datasets up to a certain size (typically around 1000 samples, 100 features, and 10 classes)."""
new_tabpfn_cells.append(nbf.v4.new_markdown_cell(markdown_tabpfn_intro))

# --- Code Cell: Train TabPFN Model ---
code_tabpfn_train = """\
# Ensure X_train_processed and y_train are available from the previous preprocessing steps
if 'X_train_processed' in locals() and 'y_train' in locals() and X_train_processed is not None and y_train is not None:
    print("Training TabPFN model...")
    
    # Initialize TabPFNClassifier
    # device='cpu' for broader compatibility. Use 'cuda' if GPU is available.
    # N_ensemble_configurations can be adjusted; 32 is a common default.
    # TabPFN has constraints on dataset size (check documentation for specifics).
    # Our dataset (sleep ID 205) is small (49 training samples after 80/20 split, ~7 features after OHE) and should fit well.
    print(f"Shape of X_train_processed for TabPFN: {X_train_processed.shape}")
    print(f"Number of unique classes in y_train for TabPFN: {y_train.nunique()}")

    if X_train_processed.shape[0] > 1000 or X_train_processed.shape[1] > 100 or y_train.nunique() > 10:
        print("Warning: The dataset dimensions might exceed TabPFN's typical optimal range (1000 samples, 100 features, 10 classes).")
        print("Performance might vary, or it might require specific configurations if using a larger pre-trained model variant.")

    tabpfn_model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    
    # Train the model
    # TabPFN expects data to be numpy arrays. X_train_processed from ColumnTransformer is typically a numpy array.
    # y_train should also be a numpy array or pandas Series.
    tabpfn_model.fit(X_train_processed, y_train)
    
    print("TabPFN model trained successfully.")
    print("Model details:", tabpfn_model)
else:
    print("Skipping TabPFN model training as X_train_processed or y_train are not available.")
    tabpfn_model = None # Ensure tabpfn_model exists even if training is skipped
"""
new_tabpfn_cells.append(nbf.v4.new_code_cell(code_tabpfn_train))

# Insert new cells after the Logistic Regression model cell
insert_position = lr_model_cell_index + 1
nb.cells = nb.cells[:insert_position] + new_tabpfn_cells + nb.cells[insert_position:]

print(f"Inserted {len(new_tabpfn_cells)} new cells for TabPFN model training after cell index {lr_model_cell_index}.")

# Write the updated notebook to a file
with open(notebook_filename, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook '{notebook_filename}' updated successfully with TabPFN model training steps.")

# Verify by listing files
print("\nFiles in current directory:")
for item in os.listdir("."):
    print(item)
