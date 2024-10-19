import numpy as np 
np.float = np
np.int = np
from river import drift as river_drift

from capymoa.classifier import HoeffdingTree
# from capymoa.drift.detectors import ADWIN
from skmultiflow.drift_detection.adwin import ADWIN as sk_adwin

def run_static(stream, grace_period=15, train_until=250):
    hoeff_tree = HoeffdingTree(schema=stream.get_schema(), grace_period=grace_period)
    preds, labels = [], []
    processed_instance = 0 
    hit = 0 
    
    while stream.has_more_instances():
        instance = stream.next_instance()
        prediction = hoeff_tree.predict(instance)
        if processed_instance <= train_until:
            hoeff_tree.train(instance)
            
        preds.append(prediction)
        labels.append(instance.y_index)
        if prediction == instance.y_index:
            hit +=1
        processed_instance += 1

    print(f"static accuracy: {hit/processed_instance}")
    return preds, labels 

def run_drift(stream, grace_period=15, retrain_size=50, trigger="scikit"):
    hoeff_tree = HoeffdingTree(schema=stream.get_schema(), grace_period=grace_period)
    
    if trigger == "scikit":
        detector = sk_adwin()
    elif trigger == "river":
        detector = river_drift.ADWIN()
    else:
        raise ValueError(f"Do not Reconigze {trigger}, shutting down drift experiment")
    
    preds, labels = [], []
    training_window = []
    processed_instance = 0
    
    drift_points = []
    hit = 0 
    has_changed = False 
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
        check_prediction = float(instance.y_index == prediction)
        hit += check_prediction
        
        if trigger == "scikit":
            old = detector.estimation
            detector.add_element(check_prediction)
            new = detector.estimation

            if detector.detected_change() and old > new:
                has_changed = True 
        
        elif trigger == "river":
            detector.update(detector) 
            if detector.drift_detected:
                has_changed = True 

        #save drift history
        drift_points.append(has_changed)

        #retrain model with last 200 instances    
        if has_changed:            
            hoeff_tree = HoeffdingTree(schema=stream.get_schema(), grace_period=grace_period)
            for instance in training_window:
                hoeff_tree.train(instance)

            print(len(training_window))
            has_changed = False 
            
    print(f"drift accuracy: {hit/processed_instance}")
    return preds, labels, drift_points
    
def run_test_then_train(stream, grace_period=15):
    hoeff_tree = HoeffdingTree(schema=stream.get_schema(), grace_period=grace_period)
    preds, labels = [], []
    hit = 0 
    processed_instance = 0 
    while stream.has_more_instances():
        instance = stream.next_instance()
        prediction = hoeff_tree.predict(instance)
        hoeff_tree.train(instance)
        
        if prediction == instance.y_index:
            hit += 1
        preds.append(prediction)
        labels.append(instance.y_index)
        processed_instance += 1
    
    print(f"test then train accuracy: {hit/processed_instance}")
    return preds, labels 