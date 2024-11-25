from river.datasets import synth
from river import tree
from river import stats, utils
from river import metrics
from river import drift as river_drift

import random 
import numpy as np
import pandas as pd 

np.float = np
np.int = np

np.random.seed(0)
random.seed(0)

from capymoa.drift.detectors import ADWIN as capy_adwin
from skmultiflow.drift_detection.adwin import ADWIN as sk_adwin

CONCEPT_SIZE  = 200

def test_on_capymoa():
    model = tree.HoeffdingTreeClassifier()

    #Acurácia prequencial em uma janela de 100 instâncias
    acc = utils.Rolling(stats.Mean(), window_size=100)
    #Acurácia global
    glob_acc = metrics.Accuracy()

    idx = 0
    detector_capymoa = capy_adwin()

    drift_points_capymoa = 0
    
    df = {
        "index" : [],
        "acc" : [],
        "glob_acc" : [],
        "drift_capymoa" : [],
    }

    window_x = []
    window_y = []
    window_size = 100 
    for c in range(3):
        dataset = synth.STAGGER(classification_function = c, seed=2)
        for x, y in dataset.take(CONCEPT_SIZE):
            pred = model.predict_one(x)
            if pred == y:
                acc.update(1)
            else:
                acc.update(0)

            glob_acc.update(pred, y)

            df["index"].append(idx)
            df["acc"].append(acc.get())
            df["glob_acc"].append(glob_acc.get())
                        
            #capymoa            
            detector_capymoa.add_element(pred == y)
            if detector_capymoa.detected_change():
                drift_points_capymoa +=1
                model = tree.HoeffdingTreeClassifier()
                for x, y in zip(window_x, window_y):
                    model.learn_one(x, y)
                df["drift_capymoa"].append(True)
            else:
                df["drift_capymoa"].append(False)
            
            #update window    
            if len(window_x) > window_size:
                window_x.pop(0)
                window_y.pop(0)
            
            window_x.append(x)
            window_y.append(y)


            model.learn_one(x,y)
            idx += 1
                
    print(f"Capymoa: {drift_points_capymoa} found")
    df = pd.DataFrame(df)
    df.to_csv("gatilhos_capymoa.csv", index=False)    

def test_on_river():
    model = tree.HoeffdingTreeClassifier()

    #Acurácia prequencial em uma janela de 100 instâncias
    acc = utils.Rolling(stats.Mean(), window_size=100)
    #Acurácia global
    glob_acc = metrics.Accuracy()

    idx = 0
    detector_river = river_drift.ADWIN()

    drift_points_river = 0
    
    df = {
        "index" : [],
        "acc" : [],
        "glob_acc" : [],
        "drift_river" : [],
    }

    window_x = []
    window_y = []
    window_size = 100 
    for c in range(3):
        dataset = synth.STAGGER(classification_function = c, seed=2)
        for x, y in dataset.take(CONCEPT_SIZE):
            pred = model.predict_one(x)
            if pred == y:
                acc.update(1)
            else:
                acc.update(0)

            glob_acc.update(pred, y)

            df["index"].append(idx)
            df["acc"].append(acc.get())
            df["glob_acc"].append(glob_acc.get())
                            
            detector_river.update(pred == y) 
            if detector_river.drift_detected:
                drift_points_river +=1
    
                model = tree.HoeffdingTreeClassifier()
                for x, y in zip(window_x, window_y):
                    model.learn_one(x, y)
    
                df["drift_river"].append(True)
            else:
                df["drift_river"].append(False)
                        
            #update window    
            if len(window_x) > window_size:
                window_x.pop(0)
                window_y.pop(0)
            
            window_x.append(x)
            window_y.append(y)


            model.learn_one(x,y)
            idx += 1
                
    print(f"River: {drift_points_river} found")
    df = pd.DataFrame(df)
    df.to_csv("gatilhos_river.csv", index=False)    

def test_on_skmultiflow():
    model = tree.HoeffdingTreeClassifier()

    #Acurácia prequencial em uma janela de 100 instâncias
    acc = utils.Rolling(stats.Mean(), window_size=100)
    #Acurácia global
    glob_acc = metrics.Accuracy()

    idx = 0
    df = {
        "index" : [],
        "acc" : [],
        "glob_acc" : [],
        "drift_skmultiflow" : [],
    }

    window_x = []
    window_y = []
    window_size = 100 
    detector_sk = sk_adwin()
    drift_points_sklearn = 0 
    for c in range(3):
        dataset = synth.STAGGER(classification_function = c, seed=2)
        for x, y in dataset.take(CONCEPT_SIZE):
            pred = model.predict_one(x)
            if pred == y:
                acc.update(1)
            else:
                acc.update(0)

            glob_acc.update(pred, y)

            df["index"].append(idx)
            df["acc"].append(acc.get())
            df["glob_acc"].append(glob_acc.get())

            old = detector_sk.estimation
            detector_sk.add_element(pred == y)
            new = detector_sk.estimation
            if detector_sk.detected_change() and old < new:
                drift_points_sklearn += 1 
        
                model = tree.HoeffdingTreeClassifier()
                for x, y in zip(window_x, window_y):
                    model.learn_one(x, y)    
                df["drift_skmultiflow"].append(True)
            else:
                df["drift_skmultiflow"].append(False)
                        
            #update window    
            if len(window_x) > window_size:
                window_x.pop(0)
                window_y.pop(0)
            
            window_x.append(x)
            window_y.append(y)


            model.learn_one(x,y)
            idx += 1
                
    print(f"Skmultiflow: {drift_points_sklearn} found")
    df = pd.DataFrame(df)
    df.to_csv("gatilhos_skmultiflow.csv", index=False)    
    
def main():
    model = tree.HoeffdingTreeClassifier()

    #Acurácia prequencial em uma janela de 100 instâncias
    acc = utils.Rolling(stats.Mean(), window_size=100)
    #Acurácia global
    glob_acc = metrics.Accuracy()

    idx = 0
    # # print("idx;preq_acc;glob_cc;drift")
    # detector_sk = sk_adwin()
    # detector_river = river_drift.ADWIN()
    # detector_capymoa = capy_adwin()
    test_on_capymoa()
    test_on_river()
    test_on_skmultiflow()
    
    # print(detector_river)
    # # for CONCEPT_SIZE in [i for i in range(100,2000,100)]:
    # drift_points_capymoa = 0
    # drift_points_river = 0 
    # drift_points_sklearn = 0 

    # df = {
    #     "index" : [],
    #     "acc" : [],
    #     "glob_acc" : [],
    #     "drift_capymoa" : [],
    #     "drift_river" : [],
    #     "drift_skmultiflow" : [],    
    # }

    # for c in range(3):
    #     dataset = synth.STAGGER(classification_function = c, seed=2)
    #     for x, y in dataset.take(CONCEPT_SIZE):
    #         pred = model.predict_one(x)
    #         if pred == y:
    #             acc.update(1)
    #         else:
    #             acc.update(0)

    #         glob_acc.update(pred, y)

    #         df["index"].append(idx)
    #         df["acc"].append(acc.get())
    #         df["glob_acc"].append(glob_acc.get())
            
    #         #capymoa            
    #         detector_capymoa.add_element(pred == y)
    #         if detector_capymoa.detected_change():
    #             drift_points_capymoa +=1
    #             df["drift_capymoa"].append(True)
    #         else:
    #             df["drift_capymoa"].append(False)
                
    #         detector_river.update(pred == y) 
    #         if detector_river.drift_detected:
    #             drift_points_river +=1
    #             df["drift_river"].append(True)
    #         else:
    #             df["drift_river"].append(False)
                
    #         #sklmultiflow            
    #         detector_sk.add_element(pred == y)
    #         if detector_sk.detected_change():
    #             drift_points_sklearn += 1 
    #             df["drift_skmultiflow"].append(True)
    #         else:
    #             df["drift_skmultiflow"].append(False)

    #         model.learn_one(x,y)
    #         idx += 1
                
    # print(f"River: {drift_points_river} found")
    # print(f"SkMultiflow: {drift_points_sklearn} found")
    # print(f"Capymoa: {drift_points_capymoa} found")
    # df = pd.DataFrame(df)
    # df.to_csv("gatilhos.csv", index=False)    


if __name__ == "__main__":
    main()
    
