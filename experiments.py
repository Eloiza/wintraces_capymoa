from capymoa.classifier import HoeffdingTree
from capymoa.drift.detectors import ADWIN

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
            #retrain model with last 200 instances
            hoeff_tree = HoeffdingTree(schema=stream.get_schema(), grace_period=grace_period)
            for instance in training_window:
                hoeff_tree.train(instance)
        
        drift_points.append(has_drifted)
    
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