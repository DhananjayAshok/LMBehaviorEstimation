from modeling import *

def pick_length(length_dict, test_dataset, strategy):
    real_lengths = np.array([length_dict[dataset] for dataset in length_dict if dataset != test_dataset])
    if strategy == "min":
        return int(real_lengths.min())
    elif strategy == "max":
        return int(real_lengths.max())
    elif strategy == "median":
        return int(np.median(real_lengths))
    else:
        raise ValueError(f"Unknown strategy {strategy}")


@click.command()
@click.option("--task", type=str, required=True)
@click.option("--model_save_name", type=str, required=True)
@click.option('--train_strategy', type=click.Choice(['min', 'max', 'median']), default="min")
@click.option('--model_kind', type=click.Choice(['linear', 'rf'], case_sensitive=False), default="linear")
@click.option('--layer', type=int, default=16)
@click.option('--label_col', type=str, default="model_label")
@click.option('--random_seed', type=int, default=42)
def main(task, model_save_name, train_strategy, model_kind, layer, label_col, random_seed):
    np.random.seed(random_seed)
    all_path = f"{results_dir}/{model_save_name}/{task}/train"
    all_datasets = os.listdir(all_path)
    all_datasets = [dataset for dataset in all_datasets if dataset not in ["buggy", "twitterfinance_l", "twitterfinance_u"]]
    length_dict = {}
    datadict = {}
    for dataset in all_datasets:
        X_train, y_train, train_df = get_xydf(task, dataset, model_save_name, "train", layer=layer, label_col=label_col, verbose=True)
        X_test, y_test, test_df = get_xydf(task, dataset, model_save_name, "test", layer=layer, label_col=label_col, verbose=True)
        datadict[dataset] = (X_train, y_train, train_df, X_test, y_test, test_df)
        length_dict[dataset] = len(X_train)
    for dataset in all_datasets:
        print(f"YYYYYYYYY Dataset: {dataset}  YYYYYYYYY")
        length = pick_length(length_dict, dataset, train_strategy)
        model = get_model(model_kind)
        _, _, _, X_test, y_test, test_df = datadict[dataset]
        X_trains = []
        y_trains = []
        train_dfs = []
        for other in all_datasets:
            if other == dataset:
                continue
            X_train, y_train, train_df, _, _, _ = datadict[other]
            if len(X_train) > length:
                idx = np.random.choice(len(X_train), length, replace=False)
                X_train = X_train[idx]
                y_train = y_train[idx]
            X_trains.append(X_train)
            y_trains.append(y_train)
            train_dfs.append(train_df)
        X_train = np.concatenate(X_trains)
        y_train = np.concatenate(y_trains)
        train_df = pd.concat(train_dfs)
        train_pred, test_pred, test_accuracy = do_model_fit(model, X_train, y_train, X_test, y_test, train_df, test_df, verbose=True, prediction_dir=None)
    return
    
if __name__ == "__main__":
    main()