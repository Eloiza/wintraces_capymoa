import glob 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D

def plot_family_evolution_wrong_x_type(df):
    #grafico do andre com bolinhas e linhas
    df['first_seen'] = df['date'].astype(str)

    sns.set_theme(style="whitegrid", palette="muted")
    ax = sns.swarmplot(data=df, x="first_seen", y="class")
    ax.set(ylabel="")

def plot_family_evolution(df, save_path="family_population_over_years.png"):
    plt.figure(figsize=(15, 8))

    df = df[(df['first_seen'].dt.year >= 2014) & (df['first_seen'].dt.year < 2022)]

    sns.set_theme(style="whitegrid", palette="muted")
    # ax = sns.swarmplot(data=df, x="first_seen", y="class", hue="class")
    
    ax = sns.stripplot(
        data=df, x="first_seen", y="class", hue="class",
        jitter=False, s=15, linewidth=0, alpha=.3,
    )
    ax.set(ylabel="")
    plt.title("Malware families population over time")
    plt.savefig(save_path)
    plt.show()

# file_names = ["results/static_tfidf512.csv", "results/test_then_train_tfidf512.csv"]
# legend_names = ["static", "test_then_train"]
# for i, filename in enumerate(file_names):
#     df = pd.read_csv(filename)
#     print(filename)
#     # legend_name = filename.split(".")[0].split("_")[-1]
#     # legend_name = filename.split(".")[0].split("_")[:]

#     plt.plot(df["index"], df["accuracy"], label=legend_names[i])

# plt.title("tfidf512 features")
# plt.legend()    
# plt.show()

# df = pd.read_csv("../new_features/csv_files/ms_defender_first_seen_labels.csv")
# df = pd.read_csv("../new_features/csv_files/wintraces.csv")

# df = df.dropna(subset=["class"])
# class_counts = df["class"].value_counts()
# df = df[df["class"].isin(class_counts[class_counts > 50].index)]

# df = df.sort_values(by=["first_seen"])
# df = df.reset_index(drop=True)
# print(f"df len: {len(df)}")
    
# sns.set_theme(style="whitegrid", palette="muted")
# ax = sns.swarmplot(data=df, x="first_seen", y="class")
# ax.set(ylabel="")

# for x_point in drift_points:
#     plt.axvline(x_point, color='black', linestyle='dashed')

# plt.show()
def plot_facetgrid_families(df, save_path="families_population_facetgrid.png"):
    # # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="class", hue="class", aspect=15, height=.8, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "first_seen",
        bw_adjust=.5, clip_on=False,
        fill=True, alpha=1, linewidth=1.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .1, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)


    g.map(label, "first_seen")

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    plt.savefig(save_path)
    plt.show()
    
def plot_model_vs_features(df, data_path, title, save_path):    
    plt.style.use("tableau-colorblind10")
    plt.figure(figsize=(15, 8))    
    for path in glob.glob(data_path):
        df = pd.read_csv(path)
        legend_name = path.split(".")[0].split("_")[-1]
        print(legend_name)
        plt.plot(df["index"], df["accuracy"], label=legend_name)

    # drift_points = [item.index() for item in df[df["drift_detected"]]]
    custom_lines = []
    if "drift_detected" in list(df.columns):
        drift_points = list(df[df["drift_detected"]].index)
        for x_point in drift_points:
            plt.axvline(x_point, color='black', linestyle='dashed')
        custom_lines.append(Line2D([0], [0], linestyle='--',color= "black", label="drift points"))
   
    plt.axvline(int(len(df) * 0.25), color='orange', linestyle="dashed")
    custom_lines.append(Line2D([0], [0], linestyle='--',color="orange", label="train samples"))

    plt.title(title)
    legend_models = plt.legend(loc=2)
    
    loc = 1 if len(custom_lines) > 1 else 0 
    legend_lines = plt.legend(handles=custom_lines, loc=loc)
    
    axes = plt.gca() 
    axes.add_artist(legend_models)
    axes.add_artist(legend_lines)
    
    plt.savefig(save_path)
    plt.show()

def main():

    # df = pd.read_csv("../new_features/csv_files/wintraces.csv")
    df = pd.read_csv("../new_features/csv_files/ms_defender_first_seen_labels.csv")
    print("len original df", len(df))
    df = df.dropna(subset=["class"])
    class_counts = df["class"].value_counts()
    
    drop_value = 50
    df = df[df["class"].isin(class_counts[class_counts > drop_value].index)]
    
    df['first_seen'] = pd.to_datetime(df['first_seen'])
    df = df.sort_values(by=["first_seen"])
    df = df.reset_index(drop=True)
    print(f"drop {drop_value} len", len(df))

    # plot_family_evolution(df)
    # plot_facetgrid_families(df)
    
    # for model_style in ["static","drift", "test_then_train"]:    
    #     plot_model_vs_features(df, f"results/default_loader/ms_defender/{model_style}/*",
    #                             f"{model_style.capitalize()} accuracy over features",
    #                             f"{model_style}_features_over.png")
        
    # plt.style.use("tableau-colorblind10")
    # plt.figure(figsize=(15, 8))    
    # feature_name = "Stationary Test"

    # path = f"./ms_defender_drop50_results/{feature_name}/static.csv"

    # path = f"stationary_test_chunk1_1k/static.csv"
    # df = pd.read_csv(path)    
    # print(path, "len(df)", len(df))
    # plt.plot(df["index"], df["accuracy"], label="static")

    # path = f"stationary_test_chunk1_1k/test_then_train.csv"
    # df = pd.read_csv(path)
    # print(path, "len(df)", len(df))
    # plt.plot(df["index"], df["accuracy"], label="test_then_train")
    
    # path = f"stationary_test_chunk1_1k/drift.csv"
    path =f"ms_defender_drop20_drift_ddm.csv"
    df = pd.read_csv(path)
    print(path, "len(df)", len(df))
    plt.plot(df["index"], df["accuracy"], label="drift")

    drift_points = list(df[df["drift"]].index)
    for x_point in drift_points:
        plt.axvline(x_point, color='black', linestyle='dashed')
        
    custom_lines = []
    custom_lines.append(Line2D([0], [0], linestyle='--',color= "black", label="drift points"))
        
    #plot train line 
    plt.axvline(int(len(df) * 0.25), color='orange', linestyle="dashed")
    custom_lines.append(Line2D([0], [0], linestyle='--',color="orange", label="train samples"))

    plt.title(f"Models Accuracy for {feature_name.upper()}")
    legend_models = plt.legend(loc=0)
    legend_lines = plt.legend(handles=custom_lines, loc=2)

    axes = plt.gca() 
    axes.add_artist(legend_models)
    axes.add_artist(legend_lines)
    
    plt.legend()
    plt.savefig(f"result_{feature_name}_static_250_chunk1.png")    
    plt.show()

if __name__ == "__main__":
    main()