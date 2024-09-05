import argparse
import numpy as np 
import pandas as pd 
import utils.utils as utils
from capymoa.stream import ARFFStream
from capymoa.classifier import HoeffdingTree
from sklearn.metrics import accuracy_score 

def main(input_file, save_dir):
    utils.create_empty_directory(save_dir)        
    stream = ARFFStream(input_file)        
    np.random.seed(0)
    
    grace_period = 15 
    hoeff_tree = HoeffdingTree(schema=stream.get_schema(), grace_period=grace_period)
    preds, labels = [], []
    
    window_instances = []
    window_size = 300 
    all_accuracy = []
    while stream.has_more_instances():        
        instance = stream.next_instance()
        window_instances.append(instance)
        if len(window_instances) < window_size:
            prediction = hoeff_tree.predict(instance)
            hoeff_tree.train(instance)

            preds.append(prediction)
            labels.append(instance.y_index)
            all_accuracy.append(accuracy_score(preds, labels))            

        if len(window_instances) == window_size:
            _ = window_instances.pop(0)
            hoeff_tree = HoeffdingTree(schema=stream.get_schema(), grace_period=grace_period)
            window_pred = []
            window_label = []
            for instance in window_instances:
                hoeff_tree.train(instance)
                hoeff_tree.predict(instance)

                window_pred.append(prediction)
                window_label.append(instance.y_index)

            all_accuracy.append(accuracy_score(window_pred, window_label))            
    
    df = pd.DataFrame({"index": [i for i in range(len(all_accuracy))], "accuracy": all_accuracy})
    df.to_csv("accuracy_window_forgetting.csv", index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments at once")
    
    parser.add_argument("--input_file", type=str, help="input file path")
    parser.add_argument("--save_dir", type=str, help="path to save results")
    
    args = parser.parse_args()
    
    main(args.input_file, args.save_dir)
