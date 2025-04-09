declare -A tasks_and_datasets=(
    ["mcqa_2_ops"]="mmlu cosmoqa piqa arc medmcqa commonsenseqa openbookqa qasc truthfulqa"
    ["unanswerable"]="qnota selfaware known_unknown"
    ["jailbreak"]="jailbreak_prompts"
    ["sentiment"]="amazonreviews yelp twitterfinance twittermteb auditorsentiment newsmtc imdb financial_phrasebank dair_emotion sst5"
    ["topic"]="nytimes bbcnews agnews"
    ["fv"]="healthver climatefever fever"
    ["toxicity"]="unintended_jigsaw jigsaw"
    ["confidence"]="naturalqa msmarco triviaqa"
    ["bullets"]="naturalqa msmarco triviaqa"
    ["json"]="naturalqa msmarco triviaqa"
    ["verbconfidence"]="naturalqa msmarco triviaqa"
)
model_name="meta-llama/Llama-3.1-8B-Instruct"
random_seed=42
random_sample_train=50000
random_sample_test=10000000
model_kind="linear"
layer=18
label_col="model_label"

source proj_params.sh
bash scripts/label_all.sh
model_save_name="${model_name#*/}"
for task in "${!tasks_and_datasets[@]}"
do
    datasets=${tasks_and_datasets[$task]}
    for dataset in $datasets
    do
        echo "XXXXXXXXXXXXXXX Running IID for $task $dataset XXXXXXXXXXXXXXXXXXX"
        middle=/$task/$dataset/
        prediction_dir=$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/predictions/$middle/
        python modeling.py --task $task --dataset $dataset --model_save_name $model_save_name --prediction_dir $prediction_dir --random_sample_train $random_sample_train --random_sample_test $random_sample_test --random_seed $random_seed --model_kind $model_kind --layer $layer --label_col $label_col
    done
done