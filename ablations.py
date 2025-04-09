import click
import pandas as pd
import numpy as np
import os
import warnings
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from models import get_model
from metrics import compute_metrics, compute_conformal_metrics
from modeling import get_xydf, results_dir, data_dir, print_base_rate


results_columns = ["random_seed", "task", "dataset", "model_save_name", "model_kind", "layer", "n_datapoints", "test_accuracy", "base_rate", "conformal_confidence", "conformal_selected", "conformal_accuracy"]
def get_results_df():
    results_df_path = results_dir + "/ablations.csv"
    if not os.path.exists(results_df_path):
        results = pd.DataFrame(columns=results_columns)
        results.to_csv(results_df_path, index=False)
    results_df = pd.read_csv(results_df_path)
    return results_df

analysis_columns = ["random_seed", "task", "dataset", "model_save_name", "base_rate", "model_correct", "probe_accuracy", "probe_selected", "model_output_tokens", "method_output_tokens", "method_correct", "fewshot_correct", "mc_pc_corr", "mec_fsc_corr", "fsc_pc_corr","diff_conf_corr", "conf_label_corr", "conf_perp_corr", "pc_perp_corr", "mc_il_corr", "pc_il_corr", "mec_il_corr", "conf_il_corr", "mc_ol_corr", "pc_ol_corr", "mec_ol_corr", "conf_ol_corr"]
def get_analysis_df():
    analysis_df_path = results_dir + "/analysis.csv"
    if not os.path.exists(analysis_df_path):
        analysis = pd.DataFrame(columns=analysis_columns)
        analysis.to_csv(analysis_df_path, index=False)
    analysis_df = pd.read_csv(analysis_df_path)
    return analysis_df


def compute_perc(array, lower, upper):
    return round((1 - ((lower <= array) & (array <= upper)).mean())*100, 2)


def do_model_fit(model, X_train, y_train, X_test, y_test, validation_split=0.15, use_confidences=False):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=1-validation_split, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split, random_state=42)
    model.fit(X_train, y_train)
    train_pred = model.predict_proba(X_train)
    val_pred = model.predict_proba(X_val)
    test_pred = model.predict_proba(X_test)
    test_acc, test_prec, test_recall, test_f1, test_auc = compute_metrics(y_test, test_pred)
    base_rate = print_base_rate(y_test)
    if not use_confidences:
        confidence = confidences[0]
        perc_selected, test_acc_conf, val_acc_conf, conf_quartile, conformal_selected = compute_conformal_metrics(y_true_val=y_val, y_pred_proba_val=val_pred, y_true_test=y_test, y_pred_proba_test=test_pred, confidence=confidence)
        return test_acc, perc_selected, test_acc_conf, base_rate, test_pred, conformal_selected
    else:
        percs_selected = []
        test_acc_confs = []
        for conf in confidences[1:]:
            perc_selected, test_acc_conf, val_acc_conf, conf_quartile, confor = compute_conformal_metrics(y_true_val=y_val, y_pred_proba_val=val_pred, y_true_test=y_test, y_pred_proba_test=test_pred, confidence=conf, quit_none=True)
            percs_selected.append(perc_selected)
            test_acc_confs.append(test_acc_conf)            
        return test_acc, percs_selected, test_acc_confs, base_rate, test_pred


def get_available_layers(task, dataset, model_save_name):
    model_respath =  f"{results_dir}/{model_save_name}/{task}/train/{dataset}"
    if not os.path.exists(model_respath):
        return []
    layers = [layer.split("_")[-1].split(".")[0] for layer in os.listdir(model_respath)]
    layers = [layer for layer in layers if layer != "states"]
    layers = [int(layer) for layer in layers]
    layers.sort()
    return layers

def safe_length(x):
    if not isinstance(x, str):
        return None
    else:
        return len(x.split())

def do_analysis(test_df):
    if "label" not in test_df.columns:
        return None
    if isinstance(test_df["label"].iloc[0], str):
        test_df["label"] = test_df["label"].apply(lambda x: ['a', 'b'].index(x)).astype(int)
    if "model_correct" not in test_df.columns:
        test_df["model_correct"] = (test_df["model_label"] == test_df["label"]).astype(int)
    model_correct = test_df["model_correct"].mean()
    test_df["probe_correct"] = (test_df["model_label"] == test_df["probe_prediction"])
    probe_correct = test_df["probe_correct"].mean()
    probe_selected = test_df["confident"].mean()
    test_df["method_prediction"] = test_df["model_label"]
    if test_df["confident"].sum() == 0:
        pass
    else:
        test_df.loc[test_df["confident"], "method_prediction"] = test_df.loc[test_df["confident"], "probe_prediction"]
    test_df["method_correct"] = (test_df["method_prediction"] == test_df["label"]).astype(int)
    method_correct = test_df["method_correct"].mean()
    if "fewshot_label" not in test_df.columns:
        fsc_pc_corr = None
        diff_conf_corr = None
        mec_fsc_corr = None
        fewshot_correct = None
    else:
        test_df["fewshot_correct"] = (test_df["label"] == test_df["fewshot_label"]).astype(int)    
        fewshot_correct = test_df["fewshot_correct"].mean()
        mec_fsc_corr = test_df["method_correct"].corr(test_df["fewshot_correct"])
        fsc_pc_corr = test_df["fewshot_correct"].corr(test_df["probe_correct"])
        test_df["difference"] = (test_df["model_label"] != test_df["fewshot_label"]).astype(int)
        if test_df["confident"].nunique() < 2:
            diff_conf_corr = None
        else:
            diff_conf_corr = test_df["confident"].corr(test_df["difference"])        
    mc_pc_corr = test_df["model_correct"].corr(test_df["probe_correct"])
    # "conf_label_corr", "conf_perp_corr", "pc_perp_corr", "mc_il_corr", "pc_il_corr", "mec_il_corr", "conf_il_corr", "mc_ol_corr", "pc_ol_corr", "mec_ol_corr", "conf_ol_corr"
    if test_df["confident"].nunique() < 2:
        conf_label_corr = None
    else:
        conf_label_corr = test_df["confident"].corr(test_df["label"])
    if "perplexity" in test_df.columns:
        if test_df["confident"].nunique() < 2:
            conf_perp_corr = None
        else:
            conf_perp_corr = test_df["confident"].corr(test_df["perplexity"])
        pc_perp_corr = test_df["probe_correct"].corr(test_df["perplexity"])
    else:
        conf_perp_corr = None
        pc_perp_corr = None
    test_df["il"] = test_df["text"].apply(safe_length)
    test_df["ol"] = test_df["output"].apply(safe_length)
    model_output_tokens = test_df['ol'].sum() # TODO: Might have to ignore nans here
    if test_df["confident"].sum() == 0:
        method_output_tokens = model_output_tokens
    else:
        method_output_tokens = test_df.loc[~test_df['confident'], "ol"].sum() + 0.5 * test_df['confident'].sum() # approx token cost maximum for every early exit
    
    mc_il_corr = test_df["model_correct"].corr(test_df["il"])
    pc_il_corr = test_df["probe_correct"].corr(test_df["il"])
    mec_il_corr = test_df["method_correct"].corr(test_df["il"])
    if test_df["confident"].nunique() < 2:
        conf_il_corr = None
    else:
        conf_il_corr = test_df["confident"].corr(test_df["il"])
    mc_ol_corr = test_df["model_correct"].corr(test_df["ol"])
    pc_ol_corr = test_df["probe_correct"].corr(test_df["ol"])
    mec_ol_corr = test_df["method_correct"].corr(test_df["ol"])
    if test_df["confident"].nunique() < 2:
        conf_ol_corr = None
    else:
        conf_ol_corr = test_df["confident"].corr(test_df["ol"])
    return [model_correct, probe_correct, probe_selected, model_output_tokens, method_output_tokens, method_correct, fewshot_correct, mc_pc_corr, mec_fsc_corr, fsc_pc_corr, diff_conf_corr, conf_label_corr, conf_perp_corr, pc_perp_corr, mc_il_corr, pc_il_corr, mec_il_corr, conf_il_corr, mc_ol_corr, pc_ol_corr, mec_ol_corr, conf_ol_corr]










fracs = [0.01, 0.02, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
confidences = [0.91, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]

@click.command()
@click.option("--task", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option('--random_seed', type=int, default=42)
@click.option('--model_kinds', default=["linear"], multiple=True)
@click.option('--label_col', type=str, default="model_label")
def main(task, dataset, random_seed, model_kinds, label_col):
    model_save_names = os.listdir(f"{results_dir}/")
    for element in model_save_names:
        if ".csv" in element:
            model_save_names.remove(element)
    np.random.seed(random_seed)
    results_df = get_results_df()
    analysis_df = get_analysis_df()
    has_results = False
    has_analysis = False
    has_rows = results_df[(results_df["task"] == task) & (results_df["dataset"] == dataset) & (results_df["random_seed"] == random_seed)].reset_index(drop=True)
    if len(has_rows) > 0:
        print(f"Already have results for {task} {dataset} with random seed {random_seed}.")
        has_results = True

    has_rows = analysis_df[(analysis_df["task"] == task) & (analysis_df["dataset"] == dataset) & (analysis_df["random_seed"] == random_seed)].reset_index(drop=True)
    if len(has_rows) > 0:
        print(f"Already have analysis for {task} {dataset} with random seed {random_seed}.")
        has_analysis = True
    if has_results and has_analysis:
        print(f"Skipping ... Remove return to avoid")
        return
    columns = results_columns
    data = []
    base_model_kind = model_kinds[0]
    model = get_model(base_model_kind)
    for model_save_name in model_save_names:
        print(f"Model: {model_save_name}")
        available_layers = get_available_layers(task, dataset, model_save_name)
        if len(available_layers) == 0:
            print(f"No available layers for {model_save_name} on {task} {dataset}")
            continue
        # base layer is the 75th percentile layer
        base_layer = available_layers[int(0.75*len(available_layers))]
        X_train, y_train, train_df = get_xydf(task, dataset, model_save_name, "train", layer=base_layer, label_col=label_col)
        X_test, y_test, test_df = get_xydf(task, dataset, model_save_name, "test", layer=base_layer, label_col=label_col)
        base_rate = print_base_rate(y_test)
        all_indices = np.arange(len(y_train))
        all_labels = np.unique(y_train)
        if len(all_labels) < 2:
            print(f"Skipping {model_save_name} on {task} {dataset} because there are less than 2 labels")
            continue
        print(f"\tDoing data scale runs")
        for frac in tqdm(fracs):
            n_points = int(frac*len(all_indices))
            if n_points < 10:
                continue
            max_retries = 10
            retries = 0
            dataok = False
            while retries < max_retries and not dataok:
                sample_indices = np.random.choice(all_indices, n_points, replace=False)
                X_train_sample = X_train[sample_indices]
                y_train_sample = y_train[sample_indices]
                for item in all_labels:
                    if item not in y_train_sample:
                        retries += 1
                        break
                if len(set(y_train_sample)) < 2:
                    retries += 1
                    continue
                dataok = True
            if dataok:
                test_accuracy, perc_selected, test_acc_conf, base_rate, test_pred, selected = do_model_fit(model, X_train_sample, y_train_sample, X_test, y_test)
                data.append([random_seed, task, dataset, model_save_name, base_model_kind, base_layer, n_points, test_accuracy, base_rate, confidences[0], perc_selected, test_acc_conf])
                if frac == 1:
                    analysis_data = [random_seed, task, dataset, model_save_name, base_rate]                 
                    if selected is None:
                        test_df["confident"] = False
                    else:
                        test_df["confident"] = selected
                    test_df["confident"] = test_df["confident"].astype(bool) 
                    test_df["probe_prediction"] = test_pred.argmax(axis=1)                    
                    analysis_results = do_analysis(test_df)
                    if analysis_results is not None:
                        for item in analysis_results:
                            analysis_data.append(item)
                        analysis_mini_df = pd.DataFrame([analysis_data], columns=analysis_columns)
                        analysis_df = pd.concat([analysis_df, analysis_mini_df], ignore_index=True)
                        analysis_df.to_csv(results_dir + "/analysis.csv", index=False)
                    test_accuracy, percs, conf_accs, base_rate, test_pred = do_model_fit(model, X_train_sample, y_train_sample, X_test, y_test, use_confidences=True)
                    for i in range(len(confidences)):
                        if i == 0:
                            continue
                        perc_selected, test_acc_conf = percs[i-1], conf_accs[i-1]
                        data.append([random_seed, task, dataset, model_save_name, base_model_kind, base_layer, n_points, test_accuracy, base_rate, confidences[i], perc_selected, test_acc_conf])
        if len(model_kinds) > 1:
            print(f"\tDoing model kind runs")
            for model_kind in tqdm(model_kinds[1:]):
                X_train, y_train, train_df = get_xydf(task, dataset, model_save_name, "train", layer=base_layer, label_col=label_col)
                X_test, y_test, test_df = get_xydf(task, dataset, model_save_name, "test", layer=base_layer, label_col=label_col)
                model = get_model(model_kind)
                test_accuracy, perc_selected, test_acc_conf, base_rate, test_pred, selected = do_model_fit(model, X_train, y_train, X_test, y_test)
                data.append([random_seed, task, dataset, model_save_name, model_kind, base_layer, len(y_train), test_accuracy, base_rate, confidences[0], perc_selected, test_acc_conf])
        model = get_model(base_model_kind)
        if len(available_layers) > 1:
            print(f"\tDoing layer runs")
            for layer in tqdm(available_layers):
                if layer == base_layer:
                    continue
                X_train, y_train, train_df = get_xydf(task, dataset, model_save_name, "train", layer=layer, label_col=label_col)
                X_test, y_test, test_df = get_xydf(task, dataset, model_save_name, "test", layer=layer, label_col=label_col)
                test_accuracy, perc_selected, test_acc_conf, base_rate, test_pred, selected = do_model_fit(model, X_train, y_train, X_test, y_test)
                data.append([random_seed, task, dataset, model_save_name, base_model_kind, layer, len(y_train), test_accuracy, base_rate, confidences[0], perc_selected, test_acc_conf])
    new_df = pd.DataFrame(data, columns=columns)
    results_df = pd.concat([results_df, new_df], ignore_index=True)
    results_df.to_csv(results_dir + "/ablations.csv", index=False)
    return
    
if __name__ == "__main__":
    main()