"""
Special code to run drift and re-extract tfidf features
"""

import os 
import numpy as np 
import pandas as pd 

from capymoa.stream import ARFFStream, Schema
from capymoa.instance import LabeledInstance
from capymoa.drift.detectors import ADWIN
from capymoa.classifier import HoeffdingTree
 
from utils.utils import load_annotations, calculate_windowed_accuracy, local_read_files_from_df, read_log
from sklearn.feature_extraction.text import TfidfVectorizer

# LOG_PATH = '../data/logs'
# ANNOTATIONS_PATH = '../data/wintraces.csv'

LOG_PATH = "../WinTracesAnalysis/wintraces/data/logs"
ANNOTATIONS_PATH = "../WinTracesAnalysis/wintraces/data/wintraces.csv"

def main():    
    max_size = 128
    #load data 
    df = load_annotations(ANNOTATIONS_PATH)
    
    train_size = int(len(df) * 0.10)
    train_df = df.iloc[:train_size]
    train_traces = local_read_files_from_df(LOG_PATH, train_df)
    print(f"Loaded {len(train_traces)} train traces")
    
    #init tfidf
    print("Training TFIDF with samples")
    vectorizer = TfidfVectorizer(stop_words=['|','\n'], max_features=max_size)
    vectorizer = vectorizer.fit(train_traces)
    
    del train_traces
    
    #create capymoa dataset schema
    schema = Schema.from_custom(feature_names=[f"att{i}" for i in range(max_size)],
                                dataset_name="tfidf_features",
                                target_attribute_name="class",
                                values_for_class_label=np.unique(df["class"].to_list()))

    #init model stuff 
    hoeff_tree = HoeffdingTree(schema=schema, grace_period=15)
    detector = ADWIN(0.002)
    
    class_map = {class_name : index for index, class_name in enumerate(np.unique(df["class"]))}
    
    labels = df["class"].to_list()
    stream_size = len(df)
    preds = []
    training_window = []
    retrain_size = 200
    processed_instance = 0
    drift_points = []
    for i, md5 in enumerate(df["md5"].to_list()):
        #load a single trace data
        path = os.path.join(LOG_PATH, md5 + ".log")
        trace_data = read_log(path)

        #extract features
        features = vectorizer.transform([trace_data])
        features = features.toarray()[0]

        #convert it to capymoa instance
        numeric_class = class_map[labels[i]]
        instance = LabeledInstance.from_array(schema, features, numeric_class)

        prediction = hoeff_tree.predict(instance)
        hoeff_tree.train(instance)
        
        preds.append(prediction)
        
        #update training window to be used in case of drift
        training_window.append(processed_instance)
        if len(training_window) == retrain_size + 1:
            training_window.pop(0)

        processed_instance += 1        
        #update adwin and test if there was a drift 
        detector.add_element(float(instance.y_index == prediction))
        has_drifted = detector.detected_change()
        if has_drifted:
            print(f"Found drift at {processed_instance}")
            hoeff_tree = HoeffdingTree(schema=schema, grace_period=15)

            print(f"Loading data to retrain tfidf!")
            train_df = df[training_window[0]:training_window[-1]]
            train_traces = local_read_files_from_df(LOG_PATH, train_df)

            print(f"Retraining tfidf!")
            vectorizer = TfidfVectorizer(stop_words=['|','\n'], max_features=max_size)
            train_data = vectorizer.fit_transform(train_traces)
    
            print(f"Retraining hoeff tree")
            for index, item in zip(training_window, train_data):
                features = item.toarray()[0]
                numeric_class = class_map[labels[index]]
                instance = LabeledInstance.from_array(schema, features, numeric_class)
                hoeff_tree.train(instance)
        
        drift_points.append(has_drifted)                
        print(f"{i}/{stream_size}")
        
        
    calculate_windowed_accuracy(preds, df["class"].to_list(), "deleteme.csv", drift_points=drift_points)

if __name__ == "__main__":
    main()