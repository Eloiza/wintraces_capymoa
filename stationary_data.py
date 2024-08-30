import numpy as np 

from statsmodels.tsa.stattools import adfuller

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
    for i in range(features.shape[1]):
        feature = features[:, i]
        adf_result = adfuller(feature)
        # kpss_result = kpss(feature, regression='c')
        print(f"Feature {i}:")
        print(f"  ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}")
        
        if adf_result[1] < 0.05:
            count_stationary_features +=1 
        
        # print(f"  KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}")
    print(f"result: {count_stationary_features}/{features.shape[1]}")

def main():
    features = np.load("ms_defender_drop50_default/tfidf_512/tfidf_512.npy")
    # labels = np.load("ms_defender_drop50_default/tfidf_512/tfidf_512_labels.npy")
    check_multivariate_stationarity(features)
    
    # window = []
    # window_size = 100
    # i = 0 
    # for feat in features:
    #     window.append(feat)
    #     if len(window) == window_size:
    #         _ = window.pop(0)
    #     mean = np.mean(window)
    #     std = np.std(window)
        
    #     print(f"{i}, {mean}, {std}")

if __name__ == "__main__":
    main()