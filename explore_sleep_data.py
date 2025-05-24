import openml
import pandas as pd

def explore_dataset():
    print("Searching for 'Health and Sleep Relation 2024' dataset on OpenML...")
    try:
        datasets_df = openml.datasets.list_datasets(output_format="dataframe")
        
        if 'name' in datasets_df.columns:
            datasets_df['name'] = datasets_df['name'].fillna('')
        else:
            print("Critical error: 'name' column not found in OpenML dataset list.")
            return

        if 'description' in datasets_df.columns:
            datasets_df['description'] = datasets_df['description'].fillna('')
        else:
            print("Warning: 'description' column not found in OpenML dataset list. Proceeding with name-based search only for filtering.")
            datasets_df['description'] = pd.Series([''] * len(datasets_df), index=datasets_df.index)


        name_match = datasets_df['name'].str.contains('sleep', case=False) & datasets_df['name'].str.contains('health', case=False)
        
        description_match = pd.Series([False] * len(datasets_df), index=datasets_df.index)
        if 'description' in datasets_df.columns: # Check if 'description' column exists
            description_match = datasets_df['description'].str.contains('sleep', case=False) & datasets_df['description'].str.contains('health', case=False)
            
        datasets_df_filtered = datasets_df[name_match | description_match].copy() # Use .copy() to avoid SettingWithCopyWarning

        dataset_id_to_try = None
        dataset_name_found = None

        if datasets_df_filtered.empty:
            print("No datasets found matching 'sleep' and 'health' keywords on OpenML.")
            print("Trying a broader search for datasets with 'sleep' in the name or description...")
            
            sleep_name_match = datasets_df['name'].str.contains('sleep', case=False)
            sleep_description_match = pd.Series([False] * len(datasets_df), index=datasets_df.index)
            if 'description' in datasets_df.columns: # Check if 'description' column exists
                 sleep_description_match = datasets_df['description'].str.contains('sleep', case=False)
            
            sleep_datasets = datasets_df[sleep_name_match | sleep_description_match].copy() # Use .copy()

            if sleep_datasets.empty:
                print("No datasets found with 'sleep' in the name or description on OpenML.")
                return
            else:
                print(f"Found {len(sleep_datasets)} datasets with 'sleep' in the name/description. Prioritizing known relevant datasets.")
                sleep_datasets.loc[:, 'NumberOfInstances'] = pd.to_numeric(sleep_datasets['NumberOfInstances'], errors='coerce').fillna(0)
                sleep_datasets.loc[:, 'NumberOfFeatures'] = pd.to_numeric(sleep_datasets['NumberOfFeatures'], errors='coerce').fillna(0)
                print(sleep_datasets[['did', 'name', 'NumberOfInstances', 'NumberOfFeatures']].head())
                
                known_sleep_ids = {
                    43800: "Sleep Heart Health Study", 
                    45068: "Sleep Regularity Study", 
                    42624: "Sleep EEG Data Set",    
                }
                
                found_known_id = False
                for ds_id, ds_name_hint in known_sleep_ids.items():
                    if ds_id in sleep_datasets['did'].values:
                        dataset_id_to_try = ds_id # This will be int
                        dataset_name_found = sleep_datasets.loc[sleep_datasets['did'] == ds_id, 'name'].iloc[0]
                        print(f"Found known relevant dataset: '{dataset_name_found}' (ID: {dataset_id_to_try}) based on hint '{ds_name_hint}'")
                        found_known_id = True
                        break
                
                if not found_known_id and not sleep_datasets.empty:
                    potential_candidates = sleep_datasets[
                        (sleep_datasets['NumberOfInstances'] > 50) & 
                        (sleep_datasets['NumberOfInstances'] < 50000) &
                        (sleep_datasets['NumberOfFeatures'] > 3) & 
                        (sleep_datasets['NumberOfFeatures'] < 100)
                    ].sort_values(by='NumberOfInstances', ascending=False)
                    
                    if not potential_candidates.empty:
                        dataset_id_to_try = potential_candidates.iloc[0]['did'] # This might be numpy.int64
                        dataset_name_found = potential_candidates.iloc[0]['name']
                        print(f"Using fallback dataset (first from broader 'sleep' search with reasonable size): '{dataset_name_found}' (ID: {dataset_id_to_try})")
                    elif not sleep_datasets.empty:
                        dataset_id_to_try = sleep_datasets.iloc[0]['did'] # This might be numpy.int64
                        dataset_name_found = sleep_datasets.iloc[0]['name']
                        print(f"Using fallback dataset (first from broader 'sleep' search, size criteria not met): '{dataset_name_found}' (ID: {dataset_id_to_try})")
                    else:
                        print("No suitable fallback dataset found in the broader 'sleep' search.")
                        return
        else:
            print(f"Found {len(datasets_df_filtered)} datasets matching 'sleep' and 'health' keywords.")
            datasets_df_filtered.loc[:, 'NumberOfInstances'] = pd.to_numeric(datasets_df_filtered['NumberOfInstances'], errors='coerce').fillna(0)
            datasets_df_filtered.loc[:, 'NumberOfFeatures'] = pd.to_numeric(datasets_df_filtered['NumberOfFeatures'], errors='coerce').fillna(0)
            
            datasets_df_filtered_sorted = datasets_df_filtered[
                (datasets_df_filtered['NumberOfInstances'] > 50) &
                (datasets_df_filtered['NumberOfFeatures'] > 3) 
            ].sort_values(by='NumberOfInstances', ascending=False)

            if not datasets_df_filtered_sorted.empty:
                print(datasets_df_filtered_sorted[['did', 'name', 'NumberOfInstances', 'NumberOfFeatures']].head())
                dataset_id_to_try = datasets_df_filtered_sorted.iloc[0]['did'] # This might be numpy.int64
                dataset_name_found = datasets_df_filtered_sorted.iloc[0]['name']
                print(f"Attempting to download dataset: '{dataset_name_found}' (ID: {dataset_id_to_try})")
            else:
                print("No datasets matching 'sleep' and 'health' keywords met the size criteria. Will try broader search if no ID selected yet.")


        if dataset_id_to_try is None:
            print("Could not identify a suitable dataset ID to download after all checks.")
            return

        # Ensure dataset_id_to_try is a Python int
        dataset_id_to_try = int(dataset_id_to_try)

        print(f"Proceeding to download dataset: '{dataset_name_found}' (ID: {dataset_id_to_try})")
        
        dataset = openml.datasets.get_dataset(
            dataset_id_to_try, 
            download_data=True, 
            download_qualities=True, 
            download_features_meta_data=True
        )
        
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format='dataframe',
            target=dataset.default_target_attribute
        )
        
        df = None
        target_column_name = dataset.default_target_attribute if dataset.default_target_attribute else 'target_variable'


        if X is None:
            print(f"Could not load data matrix X for dataset ID {dataset_id_to_try} using default target. Trying without specified target.")
            X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format='dataframe')
            if X is None:
                 print(f"Still could not load data matrix X for dataset ID {dataset_id_to_try} even without a specified target.")
                 return
            df = X.copy() 
            if y is not None: 
                if attribute_names and len(attribute_names) == len(X.columns) + 1:
                    y_name_candidates = [name for name in attribute_names if name not in X.columns.tolist()]
                    if y_name_candidates: target_column_name = y_name_candidates[0]
                df[target_column_name] = y
        elif y is not None: 
            df = X.copy() 
            df[target_column_name] = y
        else: 
            df = X.copy()

        if df is None:
            print("DataFrame could not be constructed from downloaded data.")
            return

        print(f"\nSuccessfully downloaded and loaded dataset: '{dataset.name}' (ID: {dataset.id})")
        if hasattr(dataset, 'url') and dataset.url:
            print(f"URL: {dataset.url}")
        print("--------------------------------------------------")
        print("First 5 rows of the DataFrame:")
        print(df.head())
        print("--------------------------------------------------")
        print("\nDataFrame Info:")
        df.info(verbose=True)
        print("--------------------------------------------------")

        print("\nIdentifying potential target variables (categorical/few unique values, >1 class):")
        potential_targets = []
        selected_target = None 
        for col in df.columns:
            is_categorical_col = df[col].dtype == 'object' or df[col].dtype == 'category'
            # Also consider numerical columns with few unique values as potential targets
            is_low_nunique_numeric = pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].nunique() < 20
            
            if (is_categorical_col or is_low_nunique_numeric) and df[col].nunique() > 1 and df[col].nunique() < 50 : # Target should have at least 2 classes, less than 50 for practical classification
                potential_targets.append(col)
                print(f"- '{col}' (dtype: {df[col].dtype}, unique values: {df[col].nunique()})")
        
        if not potential_targets:
            print("No obvious target variables found with current criteria for classification.")
        # Check if target_column_name (from OpenML default) is in df.columns and also in our potential_targets list
        elif target_column_name in df.columns and target_column_name in potential_targets :
            print(f"\nDefault target attribute from OpenML ('{target_column_name}') is suitable and present.")
            selected_target = target_column_name
        elif potential_targets: # If default is not suitable or not present, pick from our list
            selected_target = potential_targets[0]
            print(f"\nUsing first identified potential target: '{selected_target}'.")
        
        if selected_target:
            print(f"Value counts for '{selected_target}':")
            print(df[selected_target].value_counts(dropna=False)) # Add dropna=False for clarity
        print("--------------------------------------------------")
        
        print("\nSummary of findings:")
        print(f"- Dataset '{dataset.name}' (ID: {dataset.id}) was downloaded from OpenML.")
        print(f"- It has {df.shape[0]} rows and {df.shape[1]} columns.")
        
        current_dataset_description = dataset.description if hasattr(dataset, 'description') and dataset.description else "No description available."
        description_lines = current_dataset_description.split('\n')
        print(f"- Description: {' '.join(d for d in description_lines[:2])}...") # Print first two lines of description
        
        if selected_target:
            print(f"- A suitable target variable identified is '{selected_target}'. It has {df[selected_target].nunique()} classes.")
            print(f"- This variable is of type {df[selected_target].dtype}.")
        else:
            print("- No clear target variable for classification was identified based on common criteria.")

    except openml.exceptions.OpenMLServerException as e:
        print(f"OpenML Server Error: {e}")
        if "authentication failed" in str(e).lower() or "api key" in str(e).lower():
            print("This might be due to a missing or invalid OpenML API key or server issues.")
        elif "please provide valid dataset_id" in str(e).lower() or "Dataset not found" in str(e).lower() or "could not find dataset" in str(e).lower():
             print(f"The dataset ID {dataset_id_to_try if 'dataset_id_to_try' in locals() else 'unknown'} might be invalid or the dataset is not available/found on OpenML.")
        else:
            print("An OpenML server error occurred that was not related to authentication or dataset ID.")
    except FileNotFoundError as e: 
        print(f"File not found error during dataset processing: {e}. The dataset might be corrupted or incomplete on OpenML for ID {dataset_id_to_try if 'dataset_id_to_try' in locals() else 'unknown'}.")
    except TypeError as e: # Catching specific TypeError if it persists
        print(f"A TypeError occurred: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        print("Failed to download or process a dataset from OpenML.")

if __name__ == '__main__':
    explore_dataset()
