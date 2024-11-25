import numpy as np
np.float = np
np.int = np

from capymoa.drift.detectors import ADWIN as capy_adwin
from skmultiflow.drift_detection.adwin import ADWIN as sk_adwin
from river import drift as river_drift
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D

def draw_drift_points(points, color, linestyle):
    for x_point in points:
        plt.axvline(x_point, color=color, linestyle=linestyle)

    
def main():
    df = pd.read_csv("accuracy_window_forgetting_tfidf64_drop_banload.csv")
    print(f"Read df with size {len(df)}")
    acertos = df['acertos']
    
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

    plt.figure(figsize=(15, 8))

    #plot accuracy and drift points    
    plt.plot([i for i in range(len(df))], df['accuracy'])
    
    draw_drift_points(drift_scikit, color='red', linestyle='dashed')
    draw_drift_points(drift_river, color='blue', linestyle='dotted')
    draw_drift_points(drift_capymoa, color='green', linestyle='solid')
    
    #add line legends 
    custom_lines = []
    if len(drift_scikit) > 0:
        custom_lines.append(Line2D([0], [0], linestyle='dashed',color= "red", label="skmultiflow"))
    if len(drift_river) > 0:
        custom_lines.append(Line2D([0], [0], linestyle='dotted',color= "blue", label="river"))
    if len(drift_capymoa) > 0:
        custom_lines.append(Line2D([0], [0], linestyle='solid',color= "green", label="capymoa"))

    legend_models = plt.legend(loc=0)
    legend_lines = plt.legend(handles=custom_lines, loc=1)
    axes = plt.gca() 
    axes.add_artist(legend_models)
    axes.add_artist(legend_lines)

    plt.title("Window Forgetting Triggers w=300 TFIDF64 Drop Banload")
    plt.show()
    
if __name__ == "__main__":
    main()