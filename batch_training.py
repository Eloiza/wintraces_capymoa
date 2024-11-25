import numpy as np 
np.float = np
np.int = np

from capymoa.stream import NumpyStream
from capymoa.classifier import HoeffdingTree
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from capymoa.drift.detectors import ADWIN as capy_adwin
from skmultiflow.drift_detection.adwin import ADWIN as sk_adwin
from river import drift as river_drift
from matplotlib.lines import Line2D
import utils.utils as utils

def main():
    # features = np.load("data/ms_defender/drop15_default/tfidf_64/tfidf_64.npy")
    # labels = np.loadtxt("data/ms_defender/drop15_default/tfidf_64/tfidf_64_labels.npy", dtype="str")    

    features = np.load("./ms_defender_tfidf64_drop15_50train/tfidf_features.npy")
    labels = np.loadtxt("./ms_defender_tfidf64_drop15_50train/tfidf_features_labels.npy", dtype="str")

    # labels = labels.reshape(labels.shape[0], 1)
    # dataset = np.concatenate((features, labels), axis = 1)
    # print(f"Dataset shape: {dataset.shape}")

    print("Len labels text", len(labels))
    print("Len features: ", len(features))
    
    print(features.shape)
    print(labels.shape)
    
    unique_labels = list(np.unique(labels))
    map_labels = {unique_labels[i]:i for i in range(len(unique_labels))}
    labels_int = [map_labels[str_label] for str_label in list(labels)]
    
    TRAIN_PERC = 0.51
    
    # train_dataset= dataset[:int(len(dataset)*TRAIN_PERC)]
    # test_dataset = dataset[int(len(dataset)*TRAIN_PERC):]
    print(f"Train size: {int(len(features)*TRAIN_PERC)}")
    train_stream = NumpyStream(features[:int(len(features)*TRAIN_PERC)],
                                labels_int[:int(len(labels_int)*TRAIN_PERC)],
                                dataset_name="ms_defender_drop15_tfidf64",
                                target_type="categorical")    

    test_stream = NumpyStream(features[int(len(features)*TRAIN_PERC):],
                                labels_int[int(len(labels_int)*TRAIN_PERC):],
                                dataset_name="ms_defender_drop15_tfidf64",
                                target_type="categorical")    


    grace_period = 15
    hoeff_tree = HoeffdingTree(schema=train_stream.get_schema(), grace_period=grace_period)
    train_preds, train_labels = [], []
    processed_instance = 0 
    hit = 0 
    print("Training hoeff tree")    
    while train_stream.has_more_instances():
        instance = train_stream.next_instance()
        prediction = hoeff_tree.predict(instance)
        
        hoeff_tree.train(instance)
            
        train_preds.append(prediction)
        train_labels.append(instance.y_index)
        if prediction == instance.y_index:
            hit +=1
        processed_instance += 1

    print(f"Train accuracy: {hit/processed_instance}")
    
    processed_instance = 0 
    hit = 0
    test_accuracies = []     
    test_preds, test_labels = [], []
    print(f"Testing on hoeffding tree")
    acertos = []
    while test_stream.has_more_instances():
        instance = test_stream.next_instance()
        prediction = hoeff_tree.predict(instance)
                    
        test_preds.append(prediction)
        test_labels.append(instance.y_index)
        if prediction == instance.y_index:
            hit +=1
        processed_instance += 1
        acertos.append(prediction == instance.y_index)
        test_accuracies.append(hit/processed_instance)
        
    print(f"Test accuracy: {hit/processed_instance}")
    print("Test confusion matrix")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                               display_labels=unique_labels)
    # disp.plot(xticks_rotation="vertical")
    # disp.ax_.set_title("Matriz de Confusao Treino 25% Teste 75% Drop banload")
    # fig = disp.figure_
    # fig.set_figwidth(10)
    # fig.set_figheight(10) 
    # plt.show()
    
    adwin_scikit = sk_adwin()
    adwin_river = river_drift.ADWIN()
    adwin_capymoa = capy_adwin()
    
    drift_scikit = []
    drift_river = []
    drift_capymoa = []
    
    for i,acerto in enumerate(acertos):
        #reverse scikit learning
        old = adwin_scikit.estimation
        adwin_scikit.add_element(acerto)
        new = adwin_scikit.estimation
        if adwin_scikit.detected_change() and old > new:
            drift_scikit.append(i) 
        
        #river 
        adwin_river.update(acerto) 
        if adwin_river.drift_detected:
            drift_river.append(i) 
        
        #capymoa
        adwin_capymoa.add_element(acerto)
        if adwin_capymoa.detected_change():
            drift_capymoa.append(i)
    
    print(f"Reverse scikit multiflow: {len(drift_scikit)} - {drift_scikit}")
    print(f"River: {len(drift_river)} - {drift_river}")
    print(f"Capymoa: {len(drift_capymoa)} - {drift_capymoa}")
    
    plt.style.use("tableau-colorblind10")
    plt.figure(figsize=(15, 8))    
    plt.title("Acuracia Global 50% treino 50% teste")
    plt.plot([i for i in range(len(test_accuracies))], test_accuracies)
    

    for x_point in drift_river:
        plt.axvline(x_point, color='black', linestyle='solid')

    for x_point in drift_scikit:
        plt.axvline(x_point, color='red', linestyle='dashed')

    for x_point in drift_capymoa:
        plt.axvline(x_point, color='blue', linestyle='dotted')


    custom_lines = []
    if len(drift_river) > 0:
        custom_lines.append(Line2D([0], [0], linestyle='solid',color= "black", label="river adwin"))
    if len(drift_scikit) > 0:
        custom_lines.append(Line2D([0], [0], linestyle='dashed',color= "red", label="scikit reverse adwin"))
    if len(drift_capymoa) > 0:
        custom_lines.append(Line2D([0], [0], linestyle='dotted',color= "blue", label="capymoa adwin"))
            
    legend_models = plt.legend(loc=0)
    legend_lines = plt.legend(handles=custom_lines, loc=2)

    axes = plt.gca() 
    axes.add_artist(legend_models)
    axes.add_artist(legend_lines)

    plt.show()
    
if __name__ == "__main__":
    main()