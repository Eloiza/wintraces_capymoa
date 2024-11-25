import glob 
import argparse
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
    # plot_family_evolution(df)
    # plot_facetgrid_families(df)
    
    plt.style.use("tableau-colorblind10")
    plt.figure(figsize=(15, 8))    

    dir_name = "ms_defender_tfidf64_tfidftrain25_scikit_trigger_result"
    feature_name = "tfidf64 25% train"
    
    path = f"{dir_name}/static.csv"
    df_static = pd.read_csv(path)
    print(path, "len(df_static)", len(df_static))
    plt.plot(df_static["index"], df_static["accuracy"], label="static")

    path = f"{dir_name}/test_then_train.csv"
    df_test_then_train = pd.read_csv(path)
    print(path, "len(df_test_then_train)", len(df_test_then_train))
    plt.plot(df_test_then_train["index"], df_test_then_train["accuracy"], label="test_then_train")
    
    path = f"{dir_name}/drift.csv"
    df_drift_adwin = pd.read_csv(path)
    print(path, "len(df_drift_adwin)", len(df_drift_adwin))
    plt.plot(df_drift_adwin["index"], df_drift_adwin["accuracy"], label="drift_adwin_scikit")
    
    drift_points_adwin = list(df_drift_adwin[df_drift_adwin["drift"]].index)
    for x_point in drift_points_adwin:
        plt.axvline(x_point, color='red', linestyle='dashed')
    
    # path = "data/ms_defender/drop50_default/results/tfidf256/static.csv"
    # df_static = pd.read_csv(path)
    # print(path, "len(df_static)", len(df_static))
    # plt.plot(df_static["index"], df_static["accuracy"], label="static")

    # path = "data/ms_defender/drop50_default/results/tfidf256/test_then_train.csv"
    # df_test_then_train = pd.read_csv(path)
    # print(path, "len(df_test_then_train)", len(df_test_then_train))
    # plt.plot(df_test_then_train["index"], df_test_then_train["accuracy"], label="test_then_train")

    # path = "data/ms_defender/drop50_default/results/tfidf256/drift.csv"
    # df_drift_adwin = pd.read_csv(path)
    # print(path, "len(df_drift_adwin)", len(df_drift_adwin))
    # plt.plot(df_drift_adwin["index"], df_drift_adwin["accuracy"], label="drift_adwin")
    
    # drift_points_adwin = list(df_drift_adwin[df_drift_adwin["drift"]].index)
    # for x_point in drift_points_adwin:
    #     plt.axvline(x_point, color='red', linestyle='dashed')
    
    # path = f"ms_defender_drop50_drift_ddm_del.csv"
    # df_drift_ddm = pd.read_csv(path)
    # print(path, "len(df_drift_ddm)", len(df_drift_ddm))
    # plt.plot(df_drift_ddm["index"], df_drift_ddm["accuracy"], label="drift_ddm")

    # drift_points_ddm = list(df_drift_ddm[df_drift_ddm["drift"]].index)
    # for x_point in drift_points_ddm:
    #     plt.axvline(x_point, color='black', linestyle='dashed')
        
    custom_lines = []
    # custom_lines.append(Line2D([0], [0], linestyle='--',color= "black", label="drift ddm"))
    custom_lines.append(Line2D([0], [0], linestyle='--',color= "red", label="drift adwin scikit"))
        
    #plot train line 
    # plt.axvline(int(len(df) * 0.25), color='orange', linestyle="dashed")
    # custom_lines.append(Line2D([0], [0], linestyle='--',color="orange", label="train samples"))

    plt.title(f"Models Accuracy for {feature_name.upper()} w=32")
    
    legend_models = plt.legend(loc=0)
    legend_lines = plt.legend(handles=custom_lines, loc=2)

    # axes = plt.gca() 
    # axes.add_artist(legend_models)
    # axes.add_artist(legend_lines)
    
    plt.legend()
    plt.savefig(f"result_{feature_name}.png")    
    plt.show()

#######################################################################################
    # df = pd.read_csv("../new_features/csv_files/ms_defender_first_seen_labels.csv")
    
    # print("len original df", len(df))
    # df = df.dropna(subset=["class"])
    # class_counts = df["class"].value_counts()
    
    # drop_value = 15
    # df = df[df["class"].isin(class_counts[class_counts > drop_value].index)]
    
    # df['first_seen'] = pd.to_datetime(df['first_seen'])
    # df = df.sort_values(by=["first_seen"])
    # df = df.reset_index(drop=True)
    # print(f"drop {drop_value} len", len(df))
    
    # print(f"dataset size: {len(df)}")
    # plt.figure(figsize=(15, 8))

    # # df = df[(df['first_seen'].dt.year >= 2014) & (df['first_seen'].dt.year < 2022)]

    # sns.set_theme(style="whitegrid", palette="muted")
    # ax = sns.swarmplot(data=df, x="first_seen", y="class", hue="class")
    
    # ax = sns.stripplot(
    #     data=df, x="first_seen", y="class", hue="class",
    #     jitter=False, s=15, linewidth=0, alpha=.3,
    # )
    
    # for index in drift_points_adwin:
    #     item = df.iloc[index]
    #     plt.axvline(item["first_seen"], color='red', linestyle='dashed')
    
    # # for index in drift_points_ddm:
    # #     print(index)
    # #     item = df.iloc[index]
    # #     plt.axvline(item["first_seen"], color='black', linestyle='dashed')
    
    
    # legend_models = plt.legend(loc=0)
    # legend_lines = plt.legend(handles=custom_lines, loc=1)

    # axes = plt.gca() 
    # axes.add_artist(legend_models)
    # axes.add_artist(legend_lines)
     
    # ax.set(ylabel="")
    # plt.title("Malware families population over time with TFIDF256 Drift")
    # plt.savefig("family_population_drift_tfidf256.png")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments at once")
    
    parser.add_argument("--feature_name", type=str, help="input file path")
    parser.add_argument("--save_dir", type=str, help="path to save results")
    
    args = parser.parse_args()
    
    main()