import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


sns.set_style("whitegrid")
sns.set_context("paper")
font_size = 16
labels_font_size = 19
xtick_font_size = 19
ytick_font_size = 15
legend_font_size = 16
title_font_size = 20
scaler = 1.75
# set the font size to 14
plt.rcParams.update({'font.size': font_size * scaler})
# set xlabel font size to 16
plt.rcParams.update({'axes.labelsize': labels_font_size * scaler})
# set x tick font size to 14
plt.rcParams.update({'xtick.labelsize': xtick_font_size * scaler})
# set y tick font size to 14
plt.rcParams.update({'ytick.labelsize': ytick_font_size * scaler})
# set title font size to 20
plt.rcParams.update({'axes.titlesize': title_font_size * scaler})

colours = ["dodgerblue", "orangered", "mediumorchid"]

results_dir = os.getenv("BEHAVIOR_ANTICIPATION_RESULTS_DIR")
show = True

metrics_map = {"test_accuracy": "Total Test Consistency %", "conformal_selected": "Coverage", "conformal_confidence": "Conformal Confidence", "conformal_accuracy": "Conformal Consistency"}
task_map = {"mcqa_2_ops": "MCQA", "unanswerable": "Unanswerable Question Abstention", "topic": "Topic Classification", "sentiment": "Sentiment Analysis", "bullets": "Bullet Point Format Following", "fv": "Fact Verification", "toxicity": "Toxicity Classification" ,"confidence": "Internal Confidence Estimation", "verbconfidence": "Verbalized Confidence Estimation", "json": "JSON Format Following", "jailbreak": "Jailbreak Detection"}
metric_columns = ["test_accuracy", "conformal_selected", "conformal_accuracy"]
analysis_metrics_columns = ["base_rate", "model_correct", "probe_accuracy", "probe_selected", "model_output_tokens", "method_output_tokens", "method_correct", "fewshot_correct", "mc_pc_corr", "mec_fsc_corr", "fsc_pc_corr","diff_conf_corr", "conf_label_corr", "conf_perp_corr", "pc_perp_corr", "mc_il_corr", "pc_il_corr", "mec_il_corr", "conf_il_corr", "mc_ol_corr", "pc_ol_corr", "mec_ol_corr", "conf_ol_corr"]
conformal_confidence_default = 0.91

def do_show(save_name=None):
    if show:
        plt.show()
    elif save_name is not None:
        plt.savefig(save_name)
        plt.clf()
    else:
        print(f"Does nothing")
    return 

def get_subset_df(df, column, value):
    return df[df[column] == value].reset_index(drop=True)

# results_columns = ["random_seed", "task", "dataset", "model_save_name", "model_kind", "layer", "n_datapoints", "test_accuracy", "base_rate", "conformal_confidence", "conformal_selected", "conformal_accuracy"]
results_order = ["model_save_name", "layer", "model_kind", "n_datapoints", "conformal_confidence"]
hl_order = ["task", "dataset"]
def unroll_results_with(df, **kwargs):
    working_df = df
    for key in hl_order:
        if key in kwargs:
            if kwargs[key] is None:
                continue
            working_df = get_subset_df(working_df, key, kwargs[key])
    for column_name in results_order:
        if column_name in kwargs:
            if kwargs[column_name] is None:
                continue
            working_df = get_subset_df(working_df, column_name, kwargs[column_name])
        else:
            if column_name== "n_datapoints" and working_df['dataset'].nunique() > 1: # must handle n_datapoints
                dfs = []
                for task in working_df['task'].unique():
                    task_df = get_subset_df(working_df, 'task', task)
                    for dataset in task_df['dataset'].unique():
                        dataset_df = get_subset_df(task_df, 'dataset', dataset)
                        max_datapoints = dataset_df['n_datapoints'].max()
                        dataset_df = get_subset_df(dataset_df, 'n_datapoints', max_datapoints)
                        dfs.append(dataset_df)
                working_df = pd.concat(dfs).reset_index(drop=True)
            else:
                if column_name == "n_datapoints":
                    mode_value = working_df[column_name].max()
                elif column_name == "conformal_confidence":
                    mode_value = conformal_confidence_default
                else:
                    mode_value = working_df[column_name].mode()[0]
                print(f"Using mode value {mode_value} for column {column_name}")
                if len(get_subset_df(working_df, column_name, mode_value)) == 0:
                    breakpoint()
                working_df = get_subset_df(working_df, column_name, mode_value)
    return working_df


if results_dir is None:
    ablation_df = pd.read_csv("unfortunate/ablations.csv")
    # find the rows with jailbreak in the task column and replace it with unanswerable
    ablation_df.loc[ablation_df['task'] == "jailbreak", 'task'] = "unanswerable"
    analysis_df = pd.read_csv("unfortunate/analysis.csv")
    # find the rows with jailbreak in the task column and replace it with unanswerable
    analysis_df.loc[analysis_df['task'] == "jailbreak", 'task'] = "unanswerable"

else:
    ablation_df = pd.read_csv(results_dir + "/ablations.csv")
    analysis_df = pd.read_csv(results_dir + "/analysis.csv")

ablation_df.loc[ablation_df['dataset'] == "commonsenseqa", 'dataset'] = "csqa"
ablation_df.loc[ablation_df['dataset'] == "openbookqa", 'dataset'] = "obqa"

model_names = ablation_df['model_save_name'].unique()
# remove truthfulqa from the ablation_df
ablation_df = ablation_df[ablation_df['dataset'] != "truthfulqa"].reset_index(drop=True)
analysis_df = analysis_df[analysis_df['dataset'] != "truthfulqa"].reset_index(drop=True)
model_names = ['Llama-3.1-8B-Instruct', 'Llama-3.1-70B-Instruct', 'Llama-3.2-3B-Instruct',
 'Mistral-7B-Instruct-v0.3', 'DeepSeek-R1-Distill-Qwen-14B']
print(model_names)

def plot_stackbar():
    #max_model = ablation_df['model_save_name'].mode()[0]
    max_model = "Mistral-7B-Instruct-v0.3"
    df = unroll_results_with(ablation_df, model_save_name=max_model)
    tasks = df['task'].unique()
    for task in tasks:
        #if task != "mcqa_2_ops":
        #    continue
        df = unroll_results_with(ablation_df, task=task, model_save_name=max_model)
        # sort by dataset
        df = df.sort_values(by="dataset")        
        df = df.groupby(['dataset'])[metric_columns].mean()
        # fill the nans with 0
        df = df.fillna(0)
        df['Conformal Correct'] = df['conformal_selected'] * df['conformal_accuracy']
        df['Conformal Incorrect'] = df['conformal_selected'] * (1-df['conformal_accuracy']) 
        df['Conformal Abstain'] = 1 - df['conformal_selected']
        df = df[['Conformal Correct', 'Conformal Incorrect', 'Conformal Abstain']]
        #use_colours = ["paleturquoise", "lightsalmon", "lavender"]
        #use_colours = ["skyblue", "salmon", "plum"]
        use_colours = ["#72b5f6", "orangered", "#c385d3"]
        hatches = ['*', "X", "."]
        fig, ax = plt.subplots()
        bars1 = ax.bar(df.index, df['Conformal Correct'], label='Correct', color=use_colours[0], alpha=1)
        bars2 = ax.bar(df.index, df['Conformal Incorrect'], bottom=df['Conformal Correct'], label='Incorrect', color=use_colours[1], hatch=hatches[1], alpha=1)
        bars3 = ax.bar(df.index, df['Conformal Abstain'], bottom=df['Conformal Correct'] + df['Conformal Incorrect'], label='Abstain', color=use_colours[2], alpha=1)
        a_val=1
        circ1 = mpatches.Patch( facecolor=use_colours[0],alpha=a_val,label='Consistent')
        circ2= mpatches.Patch( facecolor=use_colours[1],alpha=a_val,hatch=hatches[1],label='Inconsistent')
        circ3 = mpatches.Patch(facecolor=use_colours[2],alpha=a_val,label='Defer')
        # for every dataset, add text of the Conformal Correct percentage in the middle of that bar, and the Conformal Incorrect at the bottom of that on            
        print(df)
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            dataset = df.index[i]
            correct_height = df.loc[dataset, 'Conformal Correct']
            textval = round(correct_height * 100, 2)
            correct_height = 0
            ax.annotate(f'{textval}',
                        xy=(bar.get_x() + bar.get_width() / 2, correct_height / 2),
                        xytext=(0, 0),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            if textval != 0:
                if correct_height - (correct_height / 2) < 0.1:
                    correct_height = df.loc[dataset, 'Conformal Correct'] + df.loc[dataset, 'Conformal Incorrect'] + 0.05
                textval = round(df.loc[dataset, 'Conformal Incorrect'] * 100, 2)
                ax.annotate(f'{textval}',
                            xy=(bar.get_x() + bar.get_width() / 2, correct_height),
                            xytext=(0, -15),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        ax.set_ylabel('Proportion of Test Set')
        #ax.set_title(f"Conformal Prediction on {task_map[task]}")
        ax.xaxis.set_tick_params(rotation=-15)
        # put legend outside the plot
        # for mcq bbox is 0.6, 1.04, upper center
        # 
        #ax.legend(handles=[circ1, circ2, circ3], loc = "upper center", bbox_to_anchor=(0.6, 1.04),
        #       frameon=True, fancybox=True, shadow=True, fontsize=legend_font_size * scaler, #bbox_to_anchor=(0.95, 0.95),
        #       bbox_transform=fig.transFigure, borderaxespad=1.0, borderpad=1.0,
        #       labelspacing=1.0)
        print(f"Showing {task}")
        do_show()
        #breakpoint()

def plot_box():
    max_model = ablation_df['model_save_name'].mode()[0]
    tasks = ablation_df['task'].unique()
    for task in tasks:
        df = unroll_results_with(ablation_df, task=task, model_save_name=max_model)
        # sort by dataset
        if task != "mcqa_2_ops":
            continue
        df = df.sort_values(by="dataset")
        for metric in metric_columns:
            if metric in ["conformal_selected", "conformal_accuracy"]:
                df[metric] = df[metric] * 100
            #if metric == "test_accuracy":
            #    df[metric] = df[metric] - df["base_rate"]
            sns.boxplot(data=df, x="dataset", y=metric, color=colours[0])
            #if "accuracy" in metric:
            #    sns.scatterplot(data=df, x="dataset", y="base_rate", color="red", label="Base Rate")
            plt.ylabel(metrics_map[metric] )#if metric != "test_accuracy" else "Accuracy - Base Rate %")
            plt.xlabel("")
            plt.xticks(rotation=-12)
            # place legend in top right
            #plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1),
            #   frameon=True, fancybox=True, shadow=True, fontsize=legend_font_size * scaler,
            #    borderaxespad=1.0, borderpad=1.0,
            #   labelspacing=1.0)
            print(f"Showing {task}")
            do_show()

def plot_layers():
    for model_name in model_names:
        df = unroll_results_with(ablation_df, model_save_name=model_name, layer=None)
        for task in df['task'].unique():
            task_df = get_subset_df(df, 'task', task)
            for dataset in task_df['dataset'].unique():
                if dataset != "amazonreviews":
                    continue
                dataset_df = get_subset_df(task_df, 'dataset', dataset)
                dataset_df['conformal_selected'].fillna(0, inplace=True)
                dataset_df["conformal_accuracy"] = dataset_df["conformal_accuracy"] * 100
                dataset_df["conformal_selected"] = dataset_df["conformal_selected"] * 100
                sns.lineplot(data=dataset_df, x="layer", y="conformal_accuracy", markers=True, label="Conformal Consistency", errorbar="ci", linestyle='--', color=colours[0], linewidth=4)
                sns.lineplot(data=dataset_df, x="layer", y="conformal_selected", markers=True, label="Coverage", errorbar="ci", linestyle='-.', color=colours[2], linewidth=4)
                sns.lineplot(data=dataset_df, x="layer", y="base_rate", markers=True, label="Base Rate", color=colours[1], linewidth=4)
                plt.title(f"{dataset}")
                plt.ylabel("Coverage / Conformal Consistency %")
                plt.xlabel("Layer")
                # hide legend
                #plt.legend().set_visible(False)# put the legend in the  center with no offset
                plt.legend(loc="center left", bbox_to_anchor=(0.5,0.5),
                           frameon=True, fancybox=True, shadow=True, fontsize=legend_font_size * scaler,
                           borderaxespad=1.0, borderpad=1.0,
                           labelspacing=1.0)
                print(f"Showing {task} {dataset}")
                do_show()

def plot_n_datapoints():
    df = unroll_results_with(ablation_df, n_datapoints=None)
    for task in df['task'].unique():
        task_df = get_subset_df(df, 'task', task)
        for dataset in task_df['dataset'].unique():
            if dataset != "mmlu":
                continue
            dataset_df = get_subset_df(task_df, 'dataset', dataset)
            dataset_df['conformal_selected'].fillna(0, inplace=True)
            dataset_df["conformal_accuracy"] = dataset_df["conformal_accuracy"] * 100
            dataset_df["conformal_selected"] = dataset_df["conformal_selected"] * 100
            sns.lineplot(data=dataset_df, x="n_datapoints", y="conformal_accuracy", markers=True, label="Conformal Consistency", errorbar="ci", linestyle='--', color=colours[0], linewidth=4)
            sns.lineplot(data=dataset_df, x="n_datapoints", y="conformal_selected", markers=True, dashes=True, label="Coverage", errorbar="ci", linestyle='-.', color=colours[2], linewidth=4)
            sns.lineplot(data=dataset_df, x="n_datapoints", y="base_rate", markers=True, label="Base Rate", color=colours[1], linewidth=4)
            #plt.title(f"{dataset} Conformal Prediction Accuracy and Coverage by Number of Datapoints")
            plt.ylabel("Conformal Consistency / Coverage %")
            #plt.ylabel("")
            plt.xlabel("Number of Datapoints")
            # hide legend
            plt.title(f"{dataset}")
            #plt.legend().set_visible(False)
            plt.legend(loc="center left", bbox_to_anchor=(0.5,0.5),
                       frameon=True, fancybox=True, shadow=True, fontsize=legend_font_size * scaler,
                       borderaxespad=1.0, borderpad=1.0,
                       labelspacing=1.0)
            print(f"Showing {task} {dataset}")
            do_show()

def plot_conformal_confidence():
    df = unroll_results_with(ablation_df, conformal_confidence=None)
    for task in df['task'].unique():
        task_df = get_subset_df(df, 'task', task)
        for dataset in task_df['dataset'].unique():
            if dataset != "unintended_jigsaw":
                continue
            dataset_df = get_subset_df(task_df, 'dataset', dataset)
            dataset_df['conformal_selected'].fillna(0, inplace=True)
            dataset_df["conformal_accuracy"] = dataset_df["conformal_accuracy"] * 100
            dataset_df["conformal_selected"] = dataset_df["conformal_selected"] * 100
            sns.lineplot(data=dataset_df, x="conformal_confidence", y="conformal_accuracy", markers=True, label="Conformal Consistency", errorbar="ci", linestyle='--', color=colours[0], linewidth=4)
            sns.lineplot(data=dataset_df, x="conformal_confidence", y="conformal_selected", markers=True, dashes=True, label="Coverage", errorbar="ci", linestyle='-.', color=colours[2], linewidth=4)
            sns.lineplot(data=dataset_df, x="conformal_confidence", y="base_rate", markers=True, label="Base Rate", color=colours[1], linewidth=4)
            plt.title(f"{dataset}")
            #plt.ylabel("Conformal Consistency / Coverage %")
            plt.ylabel("")
            plt.xlabel("Conformal Confidence")
            plt.legend().set_visible(False)
            plt.legend(loc="lower left", bbox_to_anchor=(0,0),
                       frameon=True, fancybox=True, shadow=True, fontsize=legend_font_size * scaler,
                       borderaxespad=1.0, borderpad=1.0,
                       labelspacing=1.0)
            print(f"Showing {task} {dataset}")
            do_show()




def plot_acc_efficiency():
    tasks = ["sentiment", "mcqa_2_ops", "topic", "fv", "toxicity"]
    for model_save_name in ["Llama-3.1-8B-Instruct"]:
        df = get_subset_df(analysis_df, 'model_save_name', model_save_name)
        df = df[df['task'].isin(tasks)]
        df["task"] = df["task"].map(task_map)
        df["Accuracy Loss %"] = -(df["model_correct"] - df["method_correct"]) * 100
        df["Inference Cost Reduction %"] = ((df["model_output_tokens"] - df["method_output_tokens"]) / df["model_output_tokens"]) * 100
        df = df.groupby(["task", "dataset"])[["Accuracy Loss %", "Inference Cost Reduction %"]].mean()
        df["dataset"] = df.index.get_level_values("dataset")
        df["task"] = df.index.get_level_values("task")
        sns.scatterplot(data=df, x="Inference Cost Reduction %", y="Accuracy Loss %", hue="task", style="task", s=222)
        # draw a x axis line at 0 and y axis line at 0
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        # for each row in the dataset write the dataset name right above the point:
        for i in range(len(df)):
            if df['task'].iloc[i] != "MCQA":
                continue
            h = df["Accuracy Loss %"].iloc[i] - 0.075
            w = df["Inference Cost Reduction %"].iloc[i] + 0.15
            if df["dataset"].iloc[i] == "medmcqa":
                h += 0.25
            elif df["dataset"].iloc[i] == "piqa":
                h += 0.2
            elif df["dataset"].iloc[i] == "mmlu":
                h += 0.1
            elif df["dataset"].iloc[i] == "commonsenseqa" or df["dataset"].iloc[i] == "csqa":
                h -= 0.2
            plt.text(w, h, df["dataset"].iloc[i], fontsize=font_size * scaler)
        plt.title("IID")
        plt.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0),
                   frameon=True, fancybox=True, shadow=True, fontsize=12 * scaler,
                   borderaxespad=1.0, borderpad=1.0,
                   labelspacing=0.5)
        print(f"Showing {model_save_name}")
        do_show()
        #breakpoint()
    columns=["task", "dataset", "Accuracy Loss %", "Inference Cost Reduction %"]
    data = []
    t1 = "MCQA"
    t2 = "Sentiment Analysis"
    data.append([t1, "obqa", 0.4, 67.82])
    data.append([t1, "piqa", 8.03, 76.22])
    data.append([t1, "qasc", 1.41, 61.6])
    data.append([t1, "arc", 1.06, 62.35])
    #data.append([t1, "truthfulqa", 6.34, 55.72])
    data.append([t1, "medmcqa", 0.6, 59])
    data.append([t1, "mmlu", 5.4, 65.38])
    data.append([t1, "cosmoqa", 0.1, 44.02])
    data.append([t1, "commonsenseqa", 5.11, 56.74])
    data.append([t2, "twitterfinance", 0.9, 96.05])
    data.append([t2, "dair_emotion", 1.33, 96.76])
    data.append([t2, "sst5", 2.14, 96.84])
    data.append([t2, "yelp", 0.94, 96.81])
    data.append([t2, "financial_phrasebank", 1.65, 96.12])
    data.append([t2, "auditorsentiment", -4.71, 96.35])
    data.append([t2, "imdb", -0.64, 98.04])
    data.append([t2, "twittermteb", 3.73, 96.86])
    data.append([t2, "newsmtc", 4.88, 96.76])
    data.append([t2, "amazonreviews", 9.43, 97.06])
    df = pd.DataFrame(data=data,columns=columns)
    sns.scatterplot(data=df, x="Inference Cost Reduction %", y="Accuracy Loss %", hue="task", style="task", s=222, markers=['X', 's'], palette=['orange', 'green'])
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    for i in range(len(df)):
        if df['task'].iloc[i] != "MCQA":
            continue
        h = df["Accuracy Loss %"].iloc[i] - 0.075
        w = df["Inference Cost Reduction %"].iloc[i] + 0.15
        if df["dataset"].iloc[i] == "medmcqa":
            h -= 0.15
        elif df["dataset"].iloc[i] == "piqa":
            h += 0.2
        elif df["dataset"].iloc[i] == "mmlu":
            h += 0.1
        elif df["dataset"].iloc[i] == "commonsenseqa":
            h -= 0.2
        elif df['dataset'].iloc[i] == "qasc":
            h += 0.15
        elif df['dataset'].iloc[i] == "obqa":
            h -= 0.35
        plt.text(w, h, df["dataset"].iloc[i], fontsize=font_size * scaler)
    plt.title("OOD")
    plt.legend(fontsize=20)
    plt.show()



def plot_mc_pc_correlation():
    model_df = get_subset_df(analysis_df, 'model_save_name', "Llama-3.1-8B-Instruct")
    tasks = ["sentiment", "mcqa_2_ops", "topic", "fv", "toxicity"]
    df = model_df[model_df['task'].isin(tasks)]
    df["task"] = df["task"].map(task_map)
    df = df.groupby(["task", 'dataset'])[["mc_pc_corr", 'model_correct', 'method_correct']].mean()
    sns.histplot(data=df, x="mc_pc_corr", kde=True, stat="density", color=colours[0])
    #plt.xlabel("Correlation between CoT Model Accuracy and Probe Approximation Accuracy")
    # draw a vertical line at zero and label it
    plt.axvline(0, color='black', lw=2)
    # center the showing window at 0, range it from -6 to 6
    plt.xlim(-0.6, 0.6)
    # increase the font size of the labels
    # increase size of font on y axis
    # remove x axis label
    plt.xlabel("Corr(CoT Accuracy, Approx Consistency) across datasets")
    plt.ylabel("")
    do_show()






def length_correlation(df):
    columns = ["Input Length", "Output Length"]
    index = ["Model Accuracy", "Probe Approximation Accuracy", "Probe Confidence", "Method Accuracy"]
    data = []
    mac = [df["mc_il_corr"].mean(), df["mc_ol_corr"].mean()]
    pac = [df["pc_il_corr"].mean(), df["pc_ol_corr"].mean()]
    pc = [df["conf_il_corr"].mean(), df["conf_ol_corr"].mean()]
    mea = [df["mec_il_corr"].mean(), df["mec_ol_corr"].mean()]
    data.append(mac)
    data.append(pac)
    data.append(pc)
    data.append(mea)
    plot_df = pd.DataFrame(data, columns=columns)
    plot_df.index = index
    plot_df = plot_df.loc[["Probe Approximation Consistency", "Probe Confidence"]] * 100
    sns.heatmap(plot_df, annot=True, cmap=sns.color_palette("Blues", as_cmap=True))
    plt.xticks(rotation=-10)
    plt.yticks(rotation=75)
    #plt.title("Correlation between Length and Performance Metrics")
    do_show()


def plot_length_correlation(bytask=False):
    if bytask:
        for task in analysis_df['task'].unique():
            df = get_subset_df(analysis_df, 'task', task)
            length_correlation(df)
    else:
        length_correlation(analysis_df)


def plot_confidence_correlation():
    analysis_df["task"] = analysis_df["task"].map(task_map)
    sns.boxplot(data=analysis_df, x="task", y="mc_pc_corr", color=colours[0])
    #plt.title("Correlation between Model Correctness and Probe Approximation Accuracy")
    plt.ylabel("Correlation")
    plt.xlabel("Task")
    plt.xticks(rotation=-15)
    do_show()


def plot_scale_ablation():
    models = ['Llama-3.2-3B-Instruct', 'Llama-3.1-8B-Instruct', 'Llama-3.1-70B-Instruct']
    sizes = [3, 8, 70]
    tasks_in_common = ablation_df['task'].unique()
    for model in models:
        tasks = ablation_df[ablation_df['model_save_name'] == model]['task'].unique()
        tasks_in_common = list(set(tasks_in_common).intersection(tasks))
    if len(tasks_in_common) == 0:
        print("No tasks in common between models")
        return
    columns = ["model_save_name", "task", "dataset", "random_seed", "conformal_selected", "conformal_accuracy", "test_accuracy"]
    data = []
    for task in tasks_in_common:
        datasets_in_common = ablation_df[ablation_df['task'] == task]['dataset'].unique()
        for model in models:
            datasets = ablation_df[(ablation_df['model_save_name'] == model) & (ablation_df['task'] == task)]['dataset'].unique()
            datasets_in_common = list(set(datasets_in_common).intersection(datasets))
        if len(datasets_in_common) == 0:
            print(f"No datasets in common between models for task {task}")
            continue
        for dataset in datasets_in_common:
            for model in models:
                df = unroll_results_with(ablation_df, model_save_name=model, task=task, dataset=dataset)
                for random_seed in df['random_seed'].unique():
                    seed_df = get_subset_df(df, 'random_seed', random_seed)
                    if len(seed_df) != 1:
                        print(f"Weird")
                        print(seed_df)
                        continue
                    seed_row = seed_df.iloc[0]
                    data.append(seed_row[columns])
    data_df = pd.DataFrame(data, columns=columns)
    data_df["conformal_accuracy"] = data_df["conformal_accuracy"] * 100
    data_df["Model"] = data_df["model_save_name"]
    data_df["Size"] = data_df["model_save_name"].map(dict(zip(models, sizes)))
    # sort by size
    data_df = data_df.sort_values(by="Size")
    sns.boxplot(data=data_df, x="Size", y="test_accuracy", hue="dataset")
    #plt.title(f"Model Test Accuracy by Model Size")
    plt.ylabel("Test Accuracy %")
    plt.xlabel("Model")
    plt.legend()
    do_show()


def plot_scale_ablation_layer():
    models = ['Llama-3.2-3B-Instruct', 'Llama-3.1-8B-Instruct', 'Llama-3.1-70B-Instruct']
    sizes = [3, 8, 70]
    tasks_in_common = ablation_df['task'].unique()
    for model in models:
        tasks = ablation_df[ablation_df['model_save_name'] == model]['task'].unique()
        tasks_in_common = list(set(tasks_in_common).intersection(tasks))
    if len(tasks_in_common) == 0:
        print("No tasks in common between models")
        return
    columns = ["model_save_name", "task", "dataset", "layer", "random_seed", "conformal_selected", "conformal_accuracy", "test_accuracy"]
    for task in tasks_in_common:
        datasets_in_common = ablation_df[ablation_df['task'] == task]['dataset'].unique()
        for model in models:
            datasets = ablation_df[(ablation_df['model_save_name'] == model) & (ablation_df['task'] == task)]['dataset'].unique()
            datasets_in_common = list(set(datasets_in_common).intersection(datasets))
        if len(datasets_in_common) == 0:
            print(f"No datasets in common between models for task {task}")
            continue
        for dataset in datasets_in_common:
            data = []
            for model in models:
                df = unroll_results_with(ablation_df, model_save_name=model, task=task, dataset=dataset, layer=None)
                for i in range(len(df)):
                    row = df.iloc[i]
                    data.append(row[columns])
            data_df = pd.DataFrame(data, columns=columns)
            data_df["conformal_accuracy"] = data_df["conformal_accuracy"] * 100
            data_df["Model"] = data_df["model_save_name"]
            data_df["Size"] = data_df["model_save_name"].map(dict(zip(models, sizes)))
            # sort by size
            data_df = data_df.sort_values(by="Size")
            sns.lineplot(data=data_df, x="layer", y="test_accuracy", hue="Model", errorbar="ci", style="Model", palette=colours)
            plt.title(f"{dataset}")
            plt.ylabel("")
            #plt.ylabel("Total Test Consistency %")
            plt.xlabel("Layer")
            plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0),
                       frameon=True, fancybox=True, shadow=True, fontsize=legend_font_size * scaler,
                       borderaxespad=1.0, borderpad=1.0,
                       labelspacing=0.5)
            # hide legend
            #plt.legend().set_visible(False)
            do_show()


def plot_model_kind_ablation():
    max_model = ablation_df['model_save_name'].mode()[0]
    tasks = ablation_df['task'].unique()
    df = unroll_results_with(ablation_df, model_save_name=max_model, model_kind=None)
    # do a scatterplot of conformal_selected vs conformal_accuracy with hue being model_kind and marker being task
    # group by dataset, task and model_kind and take the mean of conformal_selected and conformal_accuracy
    df = df.groupby(["dataset", "task", "model_kind"])[["conformal_selected", "conformal_accuracy"]].mean()
    # make columns of dataset, task and model_kind
    df["dataset"] = df.index.get_level_values("dataset")
    df["task"] = df.index.get_level_values("task")
    df["model_kind"] = df.index.get_level_values("model_kind")
    sns.scatterplot(data=df, x="conformal_selected", y="conformal_accuracy", hue="model_kind", style="task")
    do_show()


    





if __name__ == "__main__":
    #plot_stackbar()
    #plot_box()
    #plot_acc_efficiency()
    #plot_mc_pc_correlation()
    #plot_length_correlation()
    #plot_length_correlation(bytask=False)
    #plot_layers()
    #plot_n_datapoints()
    #plot_conformal_confidence()
    plot_scale_ablation_layer()
    #plot_scale_ablation()
    #plot_model_kind_ablation()