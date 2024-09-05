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
    features = np.load("../data/ms_defender/drop20_default/features/tfidf128/tfidf128.npy")
    labels = np.loadtxt("../data/ms_defender/drop20_default/features/tfidf128/tfidf128_labels.npy", dtype=str)
    
    class_features = {}
    for feature, class_name in zip(features, labels):
        if class_name not in class_features:
            class_features[class_name] = [] 
            
        class_features[class_name].append(feature)

    for class_name in class_features:
        similarity_matrix = cosine_similarity(class_features[class_name])
        
        num_pairs = similarity_matrix.shape[0] * (similarity_matrix.shape[0] - 1)
        pairwise_sum = np.sum(similarity_matrix) - np.trace(similarity_matrix)
        score = pairwise_sum / num_pairs

        print(f"{class_name}: {score}")


    similarity_matrix = cosine_similarity(features)
    
    num_pairs = similarity_matrix.shape[0] * (similarity_matrix.shape[0] - 1)
    pairwise_sum = np.sum(similarity_matrix) - np.trace(similarity_matrix)
    score = pairwise_sum / num_pairs
    print(f"All Score: {score}")

    
    # icc = pg.intraclass_corr(data=df, targets='Subject', raters='Rater', ratings='Score')


if __name__ == "__main__":
    main()