import os 
import csv
import sys
import pandas as pd 
import numpy as np 

from loguru import logger
from datetime import datetime
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score


def calculate_windowed_accuracy(preds, labels, save_path, drift_points=None, window_size=100):
    acc = []
    window_pred, window_label = [], [] 
    for pred, label in zip(preds, labels):
            window_pred.append(pred)
            window_label.append(label)
            acc.append(accuracy_score(window_label, window_pred)*100)
            if len(window_pred) == window_size:
                _ = window_pred.pop(0)
                _ = window_label.pop(0)
    
    if drift_points is not None:
        df = pd.DataFrame({"index":[i for i in range(len(acc))],
                           "accuracy":acc,
                           "drift": drift_points})
    else:
        df = pd.DataFrame({"index":[i for i in range(len(acc))],
                           "accuracy":acc})

    df.to_csv(save_path, index=False)

def save_as_arff(features, labels, save_path):
    labels = labels.reshape(labels.shape[0], 1)    
    dataset = np.concatenate((features, labels), axis = 1)
    
    header = [f"@ATTRIBUTE att{i} NUMERIC" for i in range(dataset.shape[1] - 1)]
    header.insert(0, "@RELATION tfidf256-features\n")
    header = "\n".join(header)
    
    class_names = ",".join([class_name for class_name in np.unique(labels)])
    header += "\n@ATTRIBUTE class "+ "{" + class_names + "}\n"
    header += "\n@DATA\n"

    text = "\n".join([",".join(dataset[i]) for i in range(len(dataset))])    
    
    with open(save_path, "w") as fp:
        fp.writelines(header + text)

def create_empty_directory(directory_path):
    if os.path.exists(directory_path):
        try:
            os.rmdir(directory_path)
            print(f"Directory '{directory_path}' removed.")
        except OSError as e:
            print(f"Error removing directory '{directory_path}': {e}")
            return
    try:
        os.makedirs(directory_path)
        print(f"Empty directory '{directory_path}' created.")
    except OSError as e:
        print(f"Error creating directory '{directory_path}': {e}")

def load_annotations(csv_path, count_drop=50):
    annotations = pd.read_csv(csv_path)
    annotations = annotations.dropna(subset=["class"])

    class_counts = annotations["class"].value_counts()
    annotations = annotations[annotations["class"].isin(class_counts[class_counts > count_drop].index)]
    
    annotations = annotations.sort_values(by=["first_seen"])
    annotations = annotations.reset_index(drop=True)
    
    return annotations

def read_and_sort(path: str) -> Dict:
    date2hash = {}
    with open(path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")
        for line in csv_reader:
            first_seen = datetime.strptime(line['first_seen'], '%Y-%m-%d %H:%M:%S')
            
            if(first_seen not in date2hash.keys()):                
                date2hash[first_seen] = []
            
            category = "unknown" if line['class'] == '' else line['class']
            date2hash[first_seen].append((line['md5'], category))

    return date2hash 

def read_log(path):
    with open(path, 'r') as trace:
        trace_data = ''
        lines = trace.readlines()
        for l in lines:
            #remove date_time, operation type (registry, process or file) and leading/trailing pipes
            idx_pipe = l.find('|') #first pipe
            idx_pipe = l.find('|',idx_pipe+1) #second pipe
            trace_data += l[idx_pipe+1:l.rfind('|')] + '\n'

    return trace_data

# def process_file_line():
    
def read_log_operation_target(path):
    with open(path, 'r') as trace:
        trace_data = ''
        lines = trace.readlines()
        for l in lines:
            l  = l.strip()
            if("WKCD_Load_Use.exe".lower() in l.lower()):
                continue
            #remove date_time, operation type (registry, process or file) and leading/trailing pipes
            
            if(len(l) < 5):
                continue 

            pipe_split = l.split("|")
            if(len(pipe_split) < 5):
                continue 
            # print("pipe split", len(pipe_split), pipe_split)
            
            if(len(pipe_split) > 7): #size 7 registry line - size 6 process
                continue 

            if(pipe_split[-1] == ''):
                try:
                    # target = pipe_split[-2].split("\\")[-2]
                    target = pipe_split[-2]
                    
                    # print("target on try", target)
                    # new_split = pipe_split[-2].split("\\") 
                    # target = new_split[-2] + "\\" +new_split[-1]
                except Exception as expt:
                    target = pipe_split[-2]
                    # print("target on except", target)
            
            else:
                target = pipe_split[-1]

            try: 
                operation = pipe_split[2]
                exe = pipe_split[4]
                # target = pipe_split[5]
                
            except Exception as expt:
                print("Excecao:")
                print("linha", l)
                print("split", pipe_split)
                print(expt)
                sys.exit(0)
            try:
                exe = exe.replace("\\Device\\HarddiskVolume2\\Windows\\System32\\", "")
                exe = exe.replace("\\Device\\HarddiskVolume2\\Monitor\\Malware\\", "")
                exe = exe.replace("\\Device\\HarddiskVolume2\\Monitor", "")
                exe = exe.replace("\\Device\\HarddiskVolume2", "")
            # \\Device\\HarddiskVolume2\\System Volume Information\\{3808876b-c176-4e48-b7ae-04046e6cc752}    
                target = target.replace("\\Device\\HarddiskVolume2", "<C:>")
                target = target.replace("\\Users\\Behemot\\", "<HOME>")
            
            except Exception as expt:
                print(expt)
            
            if("File.log" in target or "Process.log" in target):
                continue 
            
            # idx_pipe = l.find('|') #first pipe
            # idx_pipe = l.find('|',idx_pipe+1) #second pipe
            # target = l[idx_pipe+1:l.rfind('|')]
            trace_data +=  operation + "," + exe + "," + target + '\n'
            
            # print("linha", l.strip())
            # print("split", pipe_split)
            # print("result", operation + ",", exe, "," + target)
            # print()
    # print(trace_data)
    # print()
    
    return trace_data


def read_files(path: str, trace_data_path: str) -> Tuple[List[str], List[str]]:
    dict_date_to_hash = read_and_sort(trace_data_path)

    list_traces = []
    classes_list = []
    paths = []
    processed = 0
    total_len = len(dict_date_to_hash)
    for first_seen in sorted(dict_date_to_hash):
        for hash_class in dict_date_to_hash[first_seen]:
            trace_data = read_log_operation_target(path + '/' + hash_class[0] + '.log')

            list_traces.append(trace_data)
            classes_list.append(hash_class[1])
            paths.append(hash_class[0])
            
            processed += 1
            if processed % 1000 == 0:
                logger.info(f'Read: {processed}/{total_len} files')

    return list_traces, classes_list, paths

def read_files_from_df(rootdir: str, df : pd.DataFrame):
    total_len = len(df)
    list_traces = []
    for i, md5 in enumerate(df["md5"].tolist()):
        path = os.path.join(rootdir, md5 + ".log")
        trace_data = read_log_operation_target(path)
        list_traces.append(trace_data)
        
        if i % 1000 == 0:
            logger.info(f'Read: {i}/{total_len} files')

    return list_traces

def local_read_files_from_df(rootdir: str, df : pd.DataFrame):
    total_len = len(df)
    list_traces = []
    for i, md5 in enumerate(df["md5"].tolist()):
        path = os.path.join(rootdir, md5 + ".log")
        trace_data = read_log(path)
        list_traces.append(trace_data)
        
        if i % 1000 == 0:
            logger.info(f'Read: {i}/{total_len} files')

    return list_traces
 
def write_list_in_file(input_list: List[str], save_name: str) -> None:
    file_pointer = open(save_name, "w")
    for c in input_list:
        file_pointer.write(c + '\n')
    file_pointer.close()
