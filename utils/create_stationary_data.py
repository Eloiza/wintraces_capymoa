import numpy as np 
from statsmodels.tsa.stattools import adfuller


##null hyp -> the time series has a unit root aka non stationary 
##alternative -> the time series dont have a unit root so it is stationary 

# if p-value < alpha:  reject the null hyp. Accept the alternative -> is stationary 
# else: you fail to reject the null hyp. Accept the null  -> is non stationary 

from strlearn.streams import StreamGenerator

def main():
    stream = StreamGenerator(
        n_classes=2,
        n_features=512,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=105,
        n_chunks=10,
        chunk_size=1000,
        weights=[0.3, 0.7]
    )

    print(stream)
    stream.save_to_arff("stationary_test_chunk1_1200.arff")
    
if __name__ == "__main__":
    main()