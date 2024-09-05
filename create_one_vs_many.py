import re 
import numpy as np 
import argparse

def np2arff(feature_name, features, labels):
    labels = labels.reshape(labels.shape[0], 1)
    
    dataset = np.concatenate((features, labels), axis = 1)
    print(dataset.shape)
    print(dataset)
    
    header = [f"@ATTRIBUTE att{i} NUMERIC" for i in range(dataset.shape[1] - 1)]
    header.insert(0, f"@RELATION {feature_name}-features\n")
    header = "\n".join(header)
    
    class_names = ",".join([class_name for class_name in np.unique(labels)])
    header += "\n@ATTRIBUTE class "+ "{" + class_names + "}\n"
    header += "\n@DATA\n"

    text = "\n".join([",".join(dataset[i]) for i in range(len(dataset))])    
    
    with open(feature_name, "w") as fp:
        fp.writelines(header + text)


def main(input_file, label_file):
    features = np.load(input_file)
    feature_name = input_file.split("/")[-1].split("_")[0]    
    labels = np.loadtxt(label_file, dtype=str)


    for label in np.unique(labels):
        new_labels = np.array([l if l == label else "generic_malware" for l in labels])
        np2arff(f"{feature_name}_{label}_vs_many.arff", features, labels=new_labels)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple greeting program.")
    
    parser.add_argument("--feature_file", type=str, help="input file path")
    parser.add_argument("--label_file", type=str, help="input file path")
    
    args = parser.parse_args()
    
    main(args.feature_file, args.label_file)
