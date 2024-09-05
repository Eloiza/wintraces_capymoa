import os 
import numpy as np 
import pandas as pd 
import argparse

def main(path):
    static = pd.read_csv(os.path.join(path, "static.csv"))    
    print(f"mean static", np.mean(static["accuracy"]))

    test_then_train = pd.read_csv(os.path.join(path, "test_then_train.csv"))    
    print(f"mean test_then_train", np.mean(test_then_train["accuracy"]))

    drift = pd.read_csv(os.path.join(path, "drift.csv"))    
    print(f"mean drift", np.mean(drift["accuracy"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments at once")
    
    parser.add_argument("--path", type=str, help="input file path")
    
    args = parser.parse_args()
    
    main(args.path)