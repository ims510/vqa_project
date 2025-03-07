import pandas as pd
import os
import re
import numpy as np

def load_and_concatenate_chunks(directory, prefix):
    # List all files in the directory with the specified prefix
    chunk_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.pkl')]
    
    # Sort the files numerically
    chunk_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    
    return chunk_files

def flatten_and_convert_to_array(x):
    if isinstance(x, (list, tuple)):
        # Convert to numpy array and squeeze
        squeezed = np.squeeze(np.array(x))
        return squeezed
    else:
        return x

def convert_answer_to_array(x):
    if isinstance(x, list):
        return np.array(x)
    else:
        return x

def inspect_and_clean_dataframe(df):
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame size: {df.size}")
    print("DataFrame columns and example data:")
    
    for column in df.columns:
        print(f"Inspecting column: {column}")
        if column.startswith("choice") and "embedding" in column or column in ["feature_vector", "question_embedding"]:
            print(f"Applying conversion to column: {column}")
            # Apply the conversion function
            df[column] = df[column].apply(flatten_and_convert_to_array)
            print(f"Converted {column} to numpy arrays")
        elif column == "answer":
            print(f"Converting {column} to numpy arrays")
            df[column] = df[column].apply(convert_answer_to_array)
            print(f"Converted {column} to numpy arrays")

        print(f"\nColumn: {column}")
        print(f"Data type: {df[column].dtype}")
        example_data = str(df[column].iloc[0])
        print(f"Example data: {example_data[:100]}")
        # print(f"Data size: {df[column].memory_usage(deep=True)} bytes")

    return df

if __name__ == '__main__':
    directory = "data"
    prefix = "df_wide_chunk_"
    
    chunk_files = load_and_concatenate_chunks(directory, prefix)
    
    for i, chunk_file in enumerate(chunk_files):
        print(f"Loading {chunk_file}...")
        chunk = pd.read_pickle(os.path.join(directory, chunk_file))
        print(f"Loaded {chunk_file} with shape {chunk.shape}")
        
        # Inspect and clean the DataFrame
        cleaned_df = inspect_and_clean_dataframe(chunk)
        
        # Save the cleaned DataFrame
        cleaned_file_name = f"df_wide_clean_chunk_{i+1}.pkl"
        cleaned_df.to_pickle(os.path.join(directory, cleaned_file_name))
        print(f"Saved cleaned DataFrame to {cleaned_file_name}")