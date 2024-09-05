import numpy as np 
import pandas as pd 
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def check_stationarity(data):
    result = adfuller(data)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] > 0.05:
        print("Data is likely non-stationary")
    else:
        print("Data is likely stationary")
        
def check_multivariate_stationarity(features):
    count_stationary_features = 0 
    
    stationary_indexes = []
    non_stationary_indexes = []
    constant_indexes = []
    for i in range(features.shape[1]):
        feature = features[:, i]
        # print(feature)
        # print(feature.shape)
        # print()
        try:
            adf_result = adfuller(feature)
        except ValueError as err:
            # print(f"Feature {i} is constant -> all values the same. Skipped feature")
            constant_indexes.append(i)
            continue 
        
        # kpss_result = kpss(feature, regression='c')
        # print(f"Feature {i}:")
        # print(f"  ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}")
        
        if adf_result[1] < 0.05:
            count_stationary_features +=1 
            stationary_indexes.append(i)
        else:
            non_stationary_indexes.append(i)
            
        # print(f"  KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}")
    # print(f"result: {count_stationary_features}/{features.shape[1]}")
    return stationary_indexes, non_stationary_indexes, constant_indexes

def main():
    features = np.load("../data/ms_defender/drop20_default/features/tfidf128/tfidf128.npy")
    labels = np.loadtxt("../data/ms_defender/drop20_default/features/tfidf128/tfidf128_labels.npy", dtype=str)
    
    for label in np.unique(labels):
        class_features = []
        for f, l in zip(features, labels):
            if label == l:
                class_features.append(f)                
        
        class_features = np.array(class_features)
        # print(class_features.shape)
        # print(class_features)
        print(f"Analyzing class: {label} - {len(class_features)} samples") 
        stationary_indexes, non_stationary_indexes, constant_indexes = check_multivariate_stationarity(class_features)
        
        print(f"Stationary Features: {len(stationary_indexes)}")
        print(f"Non stationary Features: {len(non_stationary_indexes)}")
        print(f"Constant features: {len(constant_indexes)}")
        print()
    
    # stationary_indexes, non_stationary_indexes, constant_indexes = check_multivariate_stationarity(features)
    
    # stationary_features = features[:, stationary_indexes]

    # with open("ms_defender_drop_20_tfidf128_stationary_size125.npy", 'wb') as f:
    #     np.save(f, stationary_features)

    # with open("ms_defender_drop_20_tfidf128_stationary_size125_labels.npy", 'wb') as f:
    #     np.save(f, labels)
    
    # non_stationary_features = features[:, non_stationary_indexes]
        
    # with open("ms_defender_drop_20_tfidf128_nonstationary_size3.npy", 'wb') as f:
    #     np.save(f, non_stationary_features)
        
    # with open("ms_defender_drop_20_tfidf128_nonstationary_size3_labels.npy", 'wb') as f:
    #     np.save(f, labels)
        

if __name__ == "__main__":
    main()