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
    hidden_states_dir = f"{results_dir}/{model_save_name}/detailed/{task}/"
    df = pd.read_csv(f"{results_dir}/{model_save_name}/detailed/{task}/{dataset}_{split}_inference.csv")
    assert label_col in df.columns, f"{label_col} not in dataframe columns {df.columns} from path {results_dir}/{model_save_name}/{task}/{dataset}_{split}_inference.csv"
    X = np.load(f"{hidden_states_dir}/{split}/{dataset}/detailed_hidden_states_{layer}.npy")
    assert len(X) == len(df), f"Length of hidden states {len(X)} does not match length of dataframe {len(df)} from path {results_dir}/{model_save_name}/{task}/{dataset}_{split}_inference.csv"
    if df[label_col].isnull().sum() > 0:
        if verbose:
            print(f"Found {df[label_col].isnull().sum()} null labels in {dataset}_{split}_inference.csv with {len(df)} rows. Thats {df['model_label'].isnull().mean()*100}% of the data. Dropping these rows...")
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
    df["n_tokens"] = None
    final_X = []
    final_y = []
    token_pos = []
    token_relative_pos = []
    data_entry = []
    for i in range(len(X)):
        for j in range(len(X[i])):
            if np.isnan(X[i][j]).any():
                break
            else:
                final_X.append(X[i][j])
                final_y.append(y[i])
                token_pos.append(j)
                token_relative_pos.append(j/len(X[i]))
                data_entry.append(i)
    X = np.array(final_X)
    y = np.array(final_y)
    token_pos = np.array(token_pos)
    token_relative_pos = np.array(token_relative_pos)
    data_entry = np.array(data_entry)
    return X, y, token_pos, token_relative_pos, data_entry, df

def print_base_rate(arr, verbose=False):
    classes = range(len(set(arr)))
    class_props = []
    for class_label in classes:
        class_prop = round((arr == class_label).mean()*100, 2)
        if verbose:
            print(f"{class_label}: {class_prop}")
        class_props.append(class_prop)
    return max(class_props)

def do_model_fit(model, X_train, y_train, X_test, y_test, verbose=True):
    model.fit(X_train, y_train)
    train_pred = model.predict_proba(X_train)
    test_pred = model.predict_proba(X_test)
    test_acc, test_prec, test_recall, test_f1, test_auc = compute_metrics(y_test, test_pred)
    if verbose:
        print(f"Base Rate: ")
        print_base_rate(y_test, verbose=verbose)
        print(f"Total Test Accuracy: {test_acc}")
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
    X_train, y_train, train_token_pos, train_token_relative_pos, train_data_entry, train_df = get_xydf(task, dataset, model_save_name, "train", random_sample_train, layer=layer, label_col=label_col, verbose=True)
    X_test, y_test, test_token_pos, test_token_relative_pos, test_data_entry,  test_df = get_xydf(task, dataset, model_save_name, "test", random_sample_test,  layer=layer, label_col=label_col, verbose=True)
    model = get_model(model_kind)
    quants = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
    for q in range(len(quants)):
        if q == 0:
            continue
        prev_quant = quants[q-1]
        quant = quants[q]
        print(f"When predicting after processing in between {prev_quant} and {quant} of the tokens in the question:")
        mini_df = train_df[(train_df["token_relative_pos"] >= prev_quant) & (train_df["token_relative_pos"] < quant)]
        select_train_indices = mini_df.index
        X_train_mini = X_train[select_train_indices]
        y_train_mini = y_train[select_train_indices]
        mini_df_test = test_df[test_df["token_relative_pos"] >= prev_quant & test_df["token_relative_pos"] < quant]
        select_test_indices = mini_df_test.index
        X_test_mini = X_test[select_test_indices]
        y_test_mini = y_test[select_test_indices]
        train_pred, test_pred, test_accuracy = do_model_fit(model, X_train_mini, y_train_mini, X_test_mini, y_test_mini, verbose=True)
        confidence = test_pred.max(axis=1)
        print(f"\Confidence: {confidence.mean()}")


    columns = ["instance", "token_pos", "token_relative_pos", "model_label", "probe_prediction", "probe_confidence"]
    data = []
    for i in range(len(y_test)):
        data.append([test_data_entry[i], test_token_pos[i], test_token_relative_pos[i], y_test[i], test_pred[i], confidence[i]])
    res_df = pd.DataFrame(data, columns=columns)
    res_df["probe_correct"] = res_df["model_label"] == res_df["probe_prediction"]
    res_df.to_csv(f"{prediction_dir}/test_pred.csv", index=False)
    quants = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
    for q in range(len(quants)):
        if q == 0:
            continue
        prev_quant = quants[q-1]
        quant = quants[q]
        print(f"When predicting after processing in between {prev_quant} and {quant} of the tokens in the question:")
        mini_df = res_df[(res_df["token_relative_pos"] >= prev_quant) & (res_df["token_relative_pos"] < quant)]
        print(f"\tAccuracy: {mini_df['probe_correct'].mean()}")
        print(f"\tConfidence: {mini_df['probe_confidence'].mean()}, {mini_df['probe_confidence'].std()}")
    return
    
if __name__ == "__main__":
    main()