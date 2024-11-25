import os 
import argparse
import numpy as np 
import experiments
import utils.utils as utils 
from capymoa.stream import ARFFStream
from capymoa.stream.generator import SEA

# >>> 1888 * 0.25
# 472.0
# >>> 1888 * 0.50
# 944.0
# >>> 1888 * 0.75
# 1416.0

def main(input_file, save_dir, train_static):
    if input_file.split(".")[-1] != 'arff':
        raise ValueError(f"Invalid file extesion. Expected .arff file got {input_file.split('.')[-1]}")

    utils.create_empty_directory(save_dir)        
    stream = ARFFStream(input_file)        
    np.random.seed(0)
    
    preds_static, labels_static, acertos_static = experiments.run_static(stream, train_until=train_static)
    utils.calculate_windowed_accuracy(preds_static, labels_static, acertos_static, os.path.join(save_dir, "static.csv"))
    print("Done static")
    
    stream.restart()
    preds_test_then_train, labels_test_then_train, acertos_test_then_train = experiments.run_test_then_train(stream)
    utils.calculate_windowed_accuracy(preds_test_then_train, labels_test_then_train, acertos_test_then_train, os.path.join(save_dir, "test_then_train.csv"))
    print("Done test then train")
    
    # if not "tfidf" in input_file:        
    stream.restart()
    preds_drift, labels_drift, drift_points, acertos_drift = experiments.run_drift(stream)
    utils.calculate_windowed_accuracy(preds_drift, labels_drift, acertos_drift, os.path.join(save_dir, "drift.csv"), drift_points)
    print("Done drift")

    print("All done! :)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments at once")
    
    parser.add_argument("--input_file", type=str, help="input file path")
    parser.add_argument("--save_dir", type=str, help="path to save results")
    parser.add_argument("--train_static", type=int, help="size of stream to train static model")
    
    args = parser.parse_args()
    
    main(args.input_file, args.save_dir, args.train_static)
