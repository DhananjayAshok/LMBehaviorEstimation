import click
import pandas as pd
import numpy as np
import os
import warnings
import pickle
from models import get_model
from sklearn.model_selection import train_test_split
from metrics import compute_metrics, compute_threshold_metrics, compute_conformal_metrics

results_dir = os.getenv("BEHAVIOR_ANTICIPATION_RESULTS_DIR")
data_dir = os.getenv("BEHAVIOR_ANTICIPATION_DATA_DIR")


def get_xydf(task, dataset, model_save_name, split="train", random_sample=None, layer=16, label_col="model_label", verbose=False):
    assert split in ["train", "test"]
    hidden_states_dir = f"{results_dir}/{model_save_name}/{task}/"
    df = pd.read_csv(f"{results_dir}/{model_save_name}/{task}/{dataset}_{split}_inference.csv")
    assert label_col in df.columns, f"{label_col} not in dataframe columns {df.columns} from path {results_dir}/{model_save_name}/{task}/{dataset}_{split}_inference.csv"
    X = np.load(f"{hidden_states_dir}/{split}/{dataset}/hidden_states_{layer}.npy")
    if len(X) != len(df):
        if verbose:
            print(f"Length of hidden states {len(X)} does not match length of dataframe {len(df)}. Assuming its the first ")
        df = df.loc[:len(X)-1].reset_index(drop=True)
    assert len(X) == len(df), f"Length of hidden states {len(X)} does not match length of dataframe {len(df)} from path {results_dir}/{model_save_name}/{task}/{dataset}_{split}_inference.csv"
    if df[label_col].isnull().sum() > 0:
        if verbose:
            print(f"Found {df[label_col].isnull().sum()} null labels in {dataset}_{split}_inference.csv with {len(df)} rows. Thats {df[label_col].isnull().mean()*100}% of the data. Dropping these rows...")
    keep_indices = []
    labelna = df[label_col].isnull()
    for index in df.index:
        if not labelna[index]:
            keep_indices.append(index)
    if random_sample is None:
        random_sample = len(keep_indices)
    else:
        if random_sample <= 1.0:
            random_sample = int(random_sample*len(keep_indices))
        random_sample = min(random_sample, len(keep_indices))
    shuffled_indices = np.random.choice(keep_indices, random_sample, replace=False)
    X = X[shuffled_indices]
    df = df.iloc[shuffled_indices].reset_index(drop=True)
    df[label_col] = df[label_col].astype(int)
    y = df[label_col].values
    return X, y, df


def compute_perc(array, lower, upper):
    return round((1 - ((lower <= array) & (array <= upper)).mean())*100, 2)


def print_base_rate(arr, verbose=False):
    classes = range(len(set(arr)))
    class_props = []
    for class_label in classes:
        class_prop = round((arr == class_label).mean()*100, 2)
        if verbose:
            print(f"{class_label}: {class_prop}")
        class_props.append(class_prop)
    return max(class_props)


def safe_length(x):
    if not isinstance(x, str):
        return None
    return len(x.split(" "))


def do_model_fit(model, X_train, y_train, X_test, y_test, train_df, test_df, verbose=True, prediction_dir=None, validation_split=0.15, target_column="model_label"):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=1-validation_split, random_state=42)
    model.fit(X_train, y_train)
    train_pred = model.predict_proba(X_train)
    val_pred = model.predict_proba(X_val)
    test_pred = model.predict_proba(X_test)
    test_acc, test_prec, test_recall, test_f1, test_auc = compute_metrics(y_test, test_pred)
    if verbose:
        print(f"Base Rate: ")
        print_base_rate(y_test, verbose=verbose)
        print(f"Total Test Accuracy: {test_acc}")
        if "label" in test_df.columns:
            test_df["probe_prediction"] = test_pred.argmax(axis=1)
            test_df["model_correct"] = (test_df["model_label"] == test_df["label"]).astype(int)
            test_df["probe_correct"] = (test_df["probe_prediction"] == test_df[target_column]).astype(int)
            if target_column == "model_label":
                print(test_df.groupby("model_correct")[["probe_correct"]].mean())
                print(test_df.groupby("model_label")[["probe_correct", "model_correct"]].mean())
            else:
                print(test_df.groupby(target_column)[["probe_correct", "model_label"]].mean())
    threshold = 0.95
    perc_selected, accuracy, precision, recall, f1, auc = compute_threshold_metrics(y_test, test_pred, threshold)
    if verbose:
        print(f"With threshold {threshold}: Predicts on {round(perc_selected*100, 2)} % of samples (Test)")
        print(f"Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, AUC: {auc}")
    perc_selected, accuracy, val_acc, quartile, selected = compute_conformal_metrics(y_true_val=y_val, y_pred_proba_val=val_pred, y_true_test=y_test, y_pred_proba_test=test_pred, confidence=0.91)
    if verbose:
        if selected is None:
            print(f"Unable to find conformal quartile")
        else:
            print(f"Conformal Predicts on {round(perc_selected*100, 2)} % of samples (Test)")
            # within selected columns:
            print(f"Distribution of probe target in selected columns:")
            print_base_rate(test_df.loc[selected, target_column].values, verbose=True)
            print(f"Test Accuracy: {accuracy}, Val Accuracy: {val_acc}, Quartile: {quartile}")
            # show the recall and precision, considering the fact that the model is only predicting on a subset of the data
            # only do for binary classification
            test_df["method_selected"] = selected
            if target_column in ["false_positive", "false_negative"]:
                test_df["method_prediction"] = None
                test_df.loc[selected, "method_prediction"] = test_pred[selected].argmax(axis=1)
                test_df.loc[~selected, "method_correct"] = False
                test_df.loc[selected, "method_correct"] = (test_df.loc[selected, "method_prediction"] == test_df.loc[selected, target_column])
                true_recall = test_df.loc[(test_df[target_column] == 1), 'method_correct'].mean()
                true_precision = test_df.loc[(test_df["method_prediction"] == 1), 'method_correct'].mean()
                # How many of the 1s did we correctly identify?
                print(f"Identified {target_column} correctly (True Recall): {round(true_recall*100, 2)}%")
                # How many of our predictions of 1 were actually 0s?
                print(f"Predicted {target_column} correctly (True Precision): {round(true_precision*100, 2)}%")

            if "label" in test_df.columns and target_column == "model_label":
                test_df["method_prediction"] = test_df["model_label"]
                test_df.loc[selected, "method_prediction"] = test_pred[selected].argmax(axis=1)
                test_df["model_correct"] = (test_df["model_label"] == test_df["label"]).astype(int)
                test_df["method_correct"] = (test_df["method_prediction"] == test_df["label"]).astype(int)
                test_df["model_tokens_generated"] = test_df["output"].apply(safe_length)
                test_df["method_tokens_generated"] = test_df["output"].apply(safe_length)
                test_df.loc[selected, "method_tokens_generated"] = 0.5
                # show the model_correct vs method_correct and the model_tokens - method_tokens / model_tokens
                p1 = round((test_df['model_correct'].mean() - test_df['method_correct'].mean())*100, 2)
                p2 = round((test_df['model_tokens_generated'].sum() - test_df['method_tokens_generated'].sum()) / test_df['model_tokens_generated'].sum()*100, 2)
                print(f"Model Correct - Method Correct: {p1}%")
                print(f"Model Tokens - Method Tokens / Model Tokens: {p2}%")



        

    if prediction_dir is not None:
        np.save(f"{prediction_dir}/train_pred.npy", train_pred)
        np.save(f"{prediction_dir}/test_pred.npy", test_pred)
        model.save(f"{prediction_dir}/")
    return train_pred, test_pred, test_acc

@click.command()
@click.option("--task", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option("--model_save_name", type=str, required=True)
@click.option('--prediction_dir', type=str, default=None)
@click.option('--random_sample_train', type=int, default=None)
@click.option('--random_sample_test', type=int, default=None)
@click.option('--random_seed', type=int, default=42)
@click.option('--model_kind', type=click.Choice(['linear', 'mean'], case_sensitive=False), default="linear")
@click.option('--layer', type=int, default=16)
@click.option('--label_col', type=str, default="model_label")
def main(task, dataset, model_save_name, prediction_dir, random_sample_train, random_sample_test, random_seed, model_kind, layer, label_col):
    np.random.seed(random_seed)
    if prediction_dir is not None:
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
    X_train, y_train, train_df = get_xydf(task, dataset, model_save_name, "train", random_sample_train, layer=layer, label_col=label_col, verbose=True)
    X_test, y_test, test_df = get_xydf(task, dataset, model_save_name, "test", random_sample_test,  layer=layer, label_col=label_col, verbose=True)
    model = get_model(model_kind)
    train_pred, test_pred, test_accuracy = do_model_fit(model, X_train, y_train, X_test, y_test, train_df, test_df, verbose=True, prediction_dir=prediction_dir, target_column=label_col)
    return
    
if __name__ == "__main__":
    main()