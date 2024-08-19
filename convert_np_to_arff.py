import numpy as np 
import glob 
import argparse

def main(input_file, label_file, output_file):
    features = np.load(input_file)
    feature_name = input_file.split("/")[-1].split("_")[0]    
    labels = np.load(label_file)
    print(labels)
    labels = labels.reshape(labels.shape[0], 1)

    print("Len labels text", len(labels))
    print("Len features: ", len(features))
    
    print(features.shape)
    print(labels.shape)
    
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
    
    with open(output_file, "w") as fp:
        fp.writelines(header + text)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple greeting program.")
    
    parser.add_argument("--feature_file", type=str, help="input file path")
    parser.add_argument("--label_file", type=str, help="input file path")
    parser.add_argument("--output_file", type=str, help="path to save arff file")
    
    args = parser.parse_args()
    
    main(args.feature_file, args.label_file, args.output_file)
