declare -A tasks_and_datasets=(
    ["mcqa_2_ops"]="cosmoqa"
    ["unanswerable"]="selfaware"
    ["sentiment"]="twitterfinance"
    ["topic"]="nytimes"
)
model_name="meta-llama/Llama-3.1-8B-Instruct"
random_seed=42
random_sample_train=50000
random_sample_test=10000000
model_kind="linear"
layer=16
label_col="model_label"

source proj_params.sh
bash scripts/label_details.sh
model_save_name="${model_name#*/}"
for task in "${!tasks_and_datasets[@]}"
do
    datasets=${tasks_and_datasets[$task]}
    for dataset in $datasets
    do
        echo "XXXXXXXXXXXXXXX Running IID for $task $dataset XXXXXXXXXXXXXXXXXXX"
        middle=detailed/$task/$dataset/
        prediction_dir=$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/predictions/$middle/
        python detail_modeling.py --task $task --dataset $dataset --model_save_name $model_save_name --prediction_dir $prediction_dir --random_sample_train $random_sample_train --random_sample_test $random_sample_test --random_seed $random_seed --model_kind $model_kind --layer $layer --label_col $label_col
    done
done