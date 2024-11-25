import pandas as pd 
import matplotlib.pyplot as plt
 
def main():
    df = pd.read_csv("accuracy_window_forgetting_tfidf64.csv")
        
    plt.style.use("tableau-colorblind10")
    plt.figure(figsize=(15, 8))    
    
    # plt.plot([index + 300 for index in df["index"].to_list()], [acc*100 for acc in df["accuracy"]], label="window training")
    # plt.title("Prequential Accuracy TFIDF64 train window 300")
    # plt.xlabel("stream instance")
    # plt.ylabel("prequential accuracy")
    # plt.grid(color = 'grey', linestyle = '-', linewidth = 0.5)

    static = pd.read_csv("sea/static.csv")
    plt.plot(static["index"].to_list(), static["accuracy"], label="static")
    plt.legend()


    test_then_train = pd.read_csv("sea/test_then_train.csv")
    plt.plot(test_then_train["index"].to_list(), test_then_train["accuracy"], label="test_then_train")
    
    
    drift = pd.read_csv("sea/drift.csv")
    plt.plot(drift["index"].to_list(), drift["accuracy"], label="drift")
    
    drift_points_adwin = list(drift[drift["drift"]].index)
    for x_point in drift_points_adwin:
        plt.axvline(x_point, color='red', linestyle='dashed')

    plt.legend()
    plt.savefig("window_forgetting_tfidf64_drop15_asa.png")
    plt.show()

if __name__ == "__main__":
    main()