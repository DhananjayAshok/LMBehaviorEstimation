source proj_params.sh
model_names=("meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-70B-Instruct" "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" "mistralai/Mistral-7B-Instruct-v0.3")


declare -A standard_tasks_and_datasets=(
    ["mcqa_2_ops"]="mmlu cosmoqa piqa arc medmcqa commonsenseqa openbookqa qasc"
    ["unanswerable"]="selfaware known_unknown"
    ["jailbreak"]="wildjailbreak"
    ["sentiment"]="amazonreviews yelp twitterfinance twittermteb auditorsentiment newsmtc imdb financial_phrasebank dair_emotion sst5"
    ["fv"]="healthver climatefever fever"
    ["toxicity"]="unintended_jigsaw jigsaw"
    ["confidence"]="naturalqa msmarco triviaqa"
    ["bullets"]="naturalqa msmarco triviaqa"
    ["json"]="naturalqa msmarco triviaqa"
    ["verbconfidence"]="naturalqa msmarco triviaqa"
)


for task in "${!standard_tasks_and_datasets[@]}"
do
    datasets=${standard_tasks_and_datasets[$task]}
    for dataset in $datasets
    do
        for model_name in ${model_names[@]}
        do
            model_save_name="${model_name#*/}"
            echo XXXXXXX Doing $task $dataset $model_save_name XXXXXXX
            split="train"
            filepath="$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/$task/${dataset}_${split}_inference.csv"
            python label.py --filename $filepath --config $task
            python label.py --filename $filepath --config "label_correct" --label_col "model_correct"
            split="test"
            filepath="$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/$task/${dataset}_${split}_inference.csv"
            python label.py --filename $filepath --config $task
            python label.py --filename $filepath --config "label_correct" --label_col "model_correct"
        done
    done
done

task="topic"
datasets="nytimes bbcnews agnews"
for dataset in $datasets
do
    for model_name in ${model_names[@]}
    do
        model_save_name="${model_name#*/}"
        echo XXXXXXX Doing $task $dataset $model_save_name XXXXXXX
        split="train"
        filepath="$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/$task/${dataset}_${split}_inference.csv"
        python label.py --filename $filepath --config $task"_"$dataset
        split="test"
        filepath="$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/$task/${dataset}_${split}_inference.csv"
        python label.py --filename $filepath --config $task"_"$dataset
    done
done

