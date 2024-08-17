import os 
import numpy as np 
import pandas as pd 

from capymoa.classifier import HoeffdingTree
from capymoa.stream import ARFFStream
from capymoa.drift.detectors import ADWIN

from sklearn.metrics import accuracy_score

import argparse

import matplotlib.pyplot as plt 

def run_static(stream, grace_period=15, train_until=188):
    hoeff_tree = HoeffdingTree(schema=stream.get_schema(), grace_period=grace_period)
    preds, labels = [], []
    processed_instance = 0 
    while stream.has_more_instances():
        instance = stream.next_instance()
        prediction = hoeff_tree.predict(instance)
        if processed_instance <= train_until:
            hoeff_tree.train(instance)
            
        preds.append(prediction)
        labels.append(instance.y_index)
        processed_instance += 1
    
    return preds, labels 

def run_drift(stream, grace_period=15, retrain_size=200):
    hoeff_tree = HoeffdingTree(schema=stream.get_schema(), grace_period=grace_period)
    
    detector = ADWIN(0.002)
    preds, labels = [], []
    training_window = []
    processed_instance = 0
    drift_points = []
    
    print("Runing drift test!")
    while stream.has_more_instances():
        instance = stream.next_instance()
        prediction = hoeff_tree.predict(instance)
        hoeff_tree.train(instance)
        
        preds.append(prediction)
        labels.append(instance.y_index)
                
        #update training window to be used in case of drift
        training_window.append(instance)
        if len(training_window) == retrain_size + 1:
            training_window.pop(0)

        processed_instance += 1        
        #update adwin and test if there was a drift 
        detector.add_element(float(instance.y_index == prediction))
        has_drifted = detector.detected_change()
        if has_drifted:
            print(f"Found drift at {processed_instance}")
            #retrain model with last 200 instances
            hoeff_tree = HoeffdingTree(schema=stream.get_schema(), grace_period=grace_period)
            for instance in training_window:
                hoeff_tree.train(instance)
        
        drift_points.append(has_drifted)
            
        if not processed_instance % 100:
            print(f"Processed: {processed_instance}")
    
    return preds, labels, drift_points
    
def run_test_then_train(stream, grace_period=15):
    hoeff_tree = HoeffdingTree(schema=stream.get_schema(), grace_period=grace_period)
    preds, labels = [], []
    while stream.has_more_instances():
        instance = stream.next_instance()
        prediction = hoeff_tree.predict(instance)
        hoeff_tree.train(instance)
        
        preds.append(prediction)
        labels.append(instance.y_index)
        
    return preds, labels 

def calculate_windowed_accuracy(preds, labels, save_path, drift_points=None, window_size=100):
    acc = []
    window_pred, window_label = [], [] 
    for pred, label in zip(preds, labels):
            window_pred.append(pred)
            window_label.append(label)
            acc.append(accuracy_score(window_label, window_pred)*100)
            if len(window_pred) == window_size:
                _ = window_pred.pop(0)
                _ = window_label.pop(0)
    
    if drift_points is not None:
        df = pd.DataFrame({"accuracy":acc,
                           "drift": drift_points})
    else:
        df = pd.DataFrame({"accuracy":acc})

    df.to_csv(save_path)

def main(input_file, save_dir):
    if input_file.split(".")[-1] != '.arff':
        raise ValueError(f"Invalid file extesion. Expected .arff file got {input_file.split(".")[-1]}")
        
    stream = ARFFStream(input_file)        
    np.random.seed(0)

    preds, labels = run_static(stream)
    calculate_windowed_accuracy(preds, labels, os.path.join(save_dir, "static.csv"), drift_points)
    print("Done static")
    
    preds, labels = run_test_then_train(stream)
    calculate_windowed_accuracy(preds, labels, os.path.join(save_dir, "test_then_train.csv"), drift_points)
    print("Done test then train")
    
    preds, labels, drift_points = run_drift(stream)
    calculate_windowed_accuracy(preds, labels, os.path.join(save_dir, "drift.csv"), drift_points)
    print("Done drift")

    print("All done! :)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple greeting program.")
    
    parser.add_argument("input_file", type=str, help="input file path")
    parser.add_argument("save_dir", type=int, help="path to save results")
    
    args = parser.parse_args()
    
    main(args.input_file, args.save_dir)
