import matplotlib.pyplot as plt 
import pandas as pd 
from matplotlib.lines import Line2D


df = pd.read_csv("gatilhos.csv")

plt.figure(figsize=(15, 8))
plt.plot(df["index"].to_list(), df["acc"], label="prequential_accuracy")
plt.plot(df["index"].to_list(), df["glob_acc"], label="global_accuracy")

# print(df.iloc[0]["drift_capymoa"] == False)

for key, color in zip(["drift_capymoa", "drift_river", "drift_skmultiflow"], ["orange", "blue", "black"]):    
    drifts = [i for i in range(len(df)) if df.iloc[i][key] == True] 
    print(f"drift points {key}: ", drifts)
    for x_point in drifts:
        plt.axvline(x_point, color=color, linestyle='dashed')


custom_lines = []
custom_lines.append(Line2D([0], [0], linestyle='--',color= "orange", label="capymoa"))
custom_lines.append(Line2D([0], [0], linestyle='--',color= "blue", label="river"))
custom_lines.append(Line2D([0], [0], linestyle='--',color= "black", label="skmultiflow"))


legend_models = plt.legend(loc=0)
legend_lines = plt.legend(handles=custom_lines, loc=1)
axes = plt.gca() 
axes.add_artist(legend_models)
axes.add_artist(legend_lines)

# plt.legend()
plt.title("Triggers")
plt.show()