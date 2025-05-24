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

# Find the index of the last TabPFN model training cell
# This is typically the cell with the source containing "tabpfn_model.fit(X_train_processed, y_train)"
# or the markdown cell "## 7. TabPFN Model"
tabpfn_model_cell_index = -1
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "tabpfn_model.fit(X_train_processed, y_train)" in cell.source:
        tabpfn_model_cell_index = i
        break
    # Fallback to markdown cell if code cell structure changes slightly
    elif cell.cell_type == 'markdown' and "## 7. TabPFN Model" in cell.source:
        # If this is a markdown cell, the actual code is likely in the next cell
        if i + 1 < len(nb.cells) and nb.cells[i+1].cell_type == 'code':
            tabpfn_model_cell_index = i + 1
        else:
            tabpfn_model_cell_index = i # Or just use the markdown cell's index

if tabpfn_model_cell_index == -1:
    print("Warning: Could not find the TabPFN model training cell. Appending new cells at the end.")
    tabpfn_model_cell_index = len(nb.cells) -1 # Append at the very end if still not found


# New cells to be inserted for Model Evaluation
new_evaluation_cells = []

# --- Markdown Cell: Logistic Regression Evaluation ---
markdown_lr_eval = """\
### Logistic Regression Evaluation

We'll evaluate the performance of the trained Logistic Regression model on the test set (`X_test_processed` and `y_test`)."""
new_evaluation_cells.append(nbf.v4.new_markdown_cell(markdown_lr_eval))

# --- Code Cell: Evaluate Logistic Regression Model ---
code_lr_eval = """\
# Ensure lr_model, X_test_processed, and y_test are available
if 'lr_model' in locals() and lr_model is not None and \
   'X_test_processed' in locals() and X_test_processed is not None and \
   'y_test' in locals() and y_test is not None:
    
    print("Evaluating Logistic Regression model on the test set...")
    y_pred_lr = lr_model.predict(X_test_processed)
    
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    lr_f1_weighted = f1_score(y_test, y_pred_lr, average='weighted', zero_division=0)
    
    print(f"\\nLogistic Regression - Accuracy: {lr_accuracy:.4f}")
    print(f"Logistic Regression - F1 Score (Weighted): {lr_f1_weighted:.4f}")
    
    print("\\nLogistic Regression - Classification Report:")
    # Ensure target names are strings for the report
    class_labels_lr = np.unique(y_test).astype(str) # Use y_test for actual labels present in test set
    print(classification_report(y_test, y_pred_lr, target_names=class_labels_lr, zero_division=0))
else:
    print("Skipping Logistic Regression evaluation as the model or test data is not available.")
"""
new_evaluation_cells.append(nbf.v4.new_code_cell(code_lr_eval))


# --- Markdown Cell: TabPFN Model Evaluation ---
markdown_tabpfn_eval = """\
### TabPFN Model Evaluation

Now, let's evaluate the performance of the trained TabPFN model on the same test set."""
new_evaluation_cells.append(nbf.v4.new_markdown_cell(markdown_tabpfn_eval))

# --- Code Cell: Evaluate TabPFN Model ---
code_tabpfn_eval = """\
# Ensure tabpfn_model, X_test_processed, and y_test are available
if 'tabpfn_model' in locals() and tabpfn_model is not None and \
   'X_test_processed' in locals() and X_test_processed is not None and \
   'y_test' in locals() and y_test is not None:
    
    print("Evaluating TabPFN model on the test set...")
    y_pred_tabpfn = tabpfn_model.predict(X_test_processed)
    
    tabpfn_accuracy = accuracy_score(y_test, y_pred_tabpfn)
    tabpfn_f1_weighted = f1_score(y_test, y_pred_tabpfn, average='weighted', zero_division=0)
    
    print(f"\\nTabPFN Model - Accuracy: {tabpfn_accuracy:.4f}")
    print(f"TabPFN Model - F1 Score (Weighted): {tabpfn_f1_weighted:.4f}")
    
    print("\\nTabPFN Model - Classification Report:")
    class_labels_tabpfn = np.unique(y_test).astype(str) # Use y_test for actual labels
    print(classification_report(y_test, y_pred_tabpfn, target_names=class_labels_tabpfn, zero_division=0))
else:
    print("Skipping TabPFN model evaluation as the model or test data is not available.")
"""
new_evaluation_cells.append(nbf.v4.new_code_cell(code_tabpfn_eval))

# --- Markdown Cell: Comparison Summary ---
markdown_comparison = """\
## 8. Model Comparison Summary

Here we can briefly compare the performance of the Logistic Regression and TabPFN models based on the evaluation metrics obtained above (Accuracy and F1-score).

*(This section would typically be filled in after running the notebook and observing the actual scores. For now, it's a placeholder for that discussion.)*

- **Logistic Regression:**
  - Accuracy: [To be filled from output]
  - F1 Score (Weighted): [To be filled from output]
- **TabPFN Model:**
  - Accuracy: [To be filled from output]
  - F1 Score (Weighted): [To be filled from output]

Considerations for comparison:
- Which model performed better overall?
- Were there specific classes where one model significantly outperformed the other?
- Given TabPFN's nature (pre-trained, little tuning), how does its performance compare to the traditional, tuned (or baseline) Logistic Regression?
- For this small dataset (Sleep - ID 205), were the results as expected?
"""
new_evaluation_cells.append(nbf.v4.new_markdown_cell(markdown_comparison))


# Insert new cells after the TabPFN model training cell
insert_position = tabpfn_model_cell_index + 1
nb.cells = nb.cells[:insert_position] + new_evaluation_cells + nb.cells[insert_position:]

print(f"Inserted {len(new_evaluation_cells)} new cells for model evaluation and comparison after cell index {tabpfn_model_cell_index}.")

# Write the updated notebook to a file
with open(notebook_filename, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook '{notebook_filename}' updated successfully with model evaluation and comparison steps.")

# Verify by listing files
print("\nFiles in current directory:")
for item in os.listdir("."):
    print(item)
