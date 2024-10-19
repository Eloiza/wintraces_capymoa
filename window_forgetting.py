import argparse
import numpy as np 
import pandas as pd 
import utils.utils as utils
from capymoa.stream import ARFFStream
from capymoa.classifier import HoeffdingTree
from sklearn.metrics import accuracy_score 
from capymoa.stream import NumpyStream

def main():
    # features = np.load("data/ms_defender/drop15_default/tfidf_64/tfidf_64.npy")
    # labels = np.loadtxt("data/ms_defender/drop15_default/tfidf_64/tfidf_64_labels.npy", dtype="str")    

    features = np.load("../ms_defender_tfidf64_drop15_dropbanload/tfidf_features.npy")
    labels = np.loadtxt("../ms_defender_tfidf64_drop15_dropbanload/tfidf_features_labels.npy", dtype="str")

    print("Len labels text", len(labels))
    print("Len features: ", len(features))
    
    #convert labels to int
    unique_labels = list(np.unique(labels))
    map_labels = {unique_labels[i]:i for i in range(len(unique_labels))}
    labels_int = [map_labels[str_label] for str_label in list(labels)]
    
    stream = NumpyStream(features,
                             labels_int,
                             dataset_name="ms_defender_drop15_tfidf64",
                             target_type="categorical")    

    # stream = ARFFStream(input_file)        
    np.random.seed(0)
    
    grace_period = 15 
    
    window_instances = []
    window_size = 300 
    all_preds = []
    all_labels = []
    hit = 0 
    total = 0
    scores = [] 
    acertos = []
    while stream.has_more_instances():        
        instance = stream.next_instance()
        if len(window_instances) == window_size:
            hoeff_tree = HoeffdingTree(schema=stream.get_schema(), grace_period=grace_period)
            for ins in window_instances:
                hoeff_tree.train(ins)
            
            prediction = hoeff_tree.predict(instance)
            all_preds.append(prediction)
            all_labels.append(instance.y_index)
            scores.append(accuracy_score(all_labels, all_preds))
            if prediction == instance.y_index:
                hit += 1
            total += 1
            _ = window_instances.pop(0)
            
            acertos.append(instance.y_index == prediction)
        window_instances.append(instance)
    
    print("Total accuracy:", accuracy_score(all_labels, all_preds))    
    print(len(scores))
    df = pd.DataFrame({"index": [i for i in range(len(scores))], "accuracy": scores, "acertos":acertos})
    df.to_csv("accuracy_window_forgetting_tfidf64_drop_banload.csv", index=False)
    print(len(df))
    
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run experiments at once")
    
    # parser.add_argument("--input_file", type=str, help="input file path")
    # 
    # args = parser.parse_args()
    
    # main(args.input_file)
    main()