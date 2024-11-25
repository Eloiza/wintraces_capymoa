import pandas as pd 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

def avg_intra_class_similarity(similarity_matrix):
    # Sum of all pairwise similarities excluding diagonal
    pairwise_sum = np.sum(similarity_matrix) - np.trace(similarity_matrix)
    # Number of pairs
    num_pairs = similarity_matrix.shape[0] * (similarity_matrix.shape[0] - 1)
    return pairwise_sum / num_pairs

def main():
    feature_size= [16, 32, 64, 128]
    for size in feature_size:
        print(f"TFIDF {size}")
        features = np.load(f"../data/ms_defender/drop15_default/tfidf_{size}/tfidf_{size}.npy")
        labels = np.loadtxt(f"../data/ms_defender/drop15_default/tfidf_{size}/tfidf_{size}_labels.npy", dtype=str)
        
        class_features = {}
        for feature, class_name in zip(features, labels):
            if class_name not in class_features:
                class_features[class_name] = [] 
                
            class_features[class_name].append(feature)

        for class_name in class_features:
            similarity_matrix = cosine_similarity(class_features[class_name])
            
            num_pairs = similarity_matrix.shape[0] * similarity_matrix.shape[0]
            pairwise_sum = np.sum(similarity_matrix) - np.trace(similarity_matrix)
            # pairwise_sum = np.sum(similarity_matrix)
            score = pairwise_sum / num_pairs

        
            print(f"{class_name}: {score}")
         
        
        num_pairs = similarity_matrix.shape[0] * (similarity_matrix.shape[0] - 1)
        pairwise_sum = np.sum(similarity_matrix) - np.trace(similarity_matrix)
        score = pairwise_sum / num_pairs
        print(f"Feature size {size} all score: {score}")
        print()
    
    # icc = pg.intraclass_corr(data=df, targets='Subject', raters='Rater', ratings='Score')


if __name__ == "__main__":
    main()