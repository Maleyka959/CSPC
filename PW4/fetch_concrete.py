"""
Fetch the Concrete Compressive Strength dataset from the UCI Machine Learning Repository,
rename columns as specified, print metadata/variables, and save to data/concrete.csv.
"""
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def fetch_and_save_concrete():
    print("Fetching Concrete Compressive Strength dataset from UCI...")
    concrete = fetch_ucirepo(id=165)
    
    # Print metadata and variables table as required by Task A1
    print("\n--- METADATA ---")
    print(concrete.metadata)
    
    print("\n--- VARIABLES TABLE ---")
    print(concrete.variables)
    
    # Extract features and targets
    X = concrete.data.features
    y = concrete.data.targets
    
    # Rename columns by position to match the exact lab specification
    X = X.copy()
    X.columns = [
        "cement", "slag", "ash", "water", "superplastic",
        "coarse_agg", "fine_agg", "age"
    ]
    y = y.iloc[:, 0].rename("strength")
    
    # Create 'data' directory if it doesn't exist, then save CSV
    os.makedirs("data", exist_ok=True)
    output_path = "data/concrete.csv"
    
    combined_df = X.assign(strength=y)
    combined_df.to_csv(output_path, index=False)
    print(f"\nDataset successfully saved to {output_path}")
    print(f"Dataset shape: {combined_df.shape}")

if __name__ == "__main__":
    fetch_and_save_concrete()