declare -A tasks_and_datasets=(
    #["mcqa_2_ops"]="mmlu cosmoqa piqa arc medmcqa commonsenseqa openbookqa qasc truthfulqa"
    #["unanswerable"]="selfaware known_unknown"
    #["jailbreak"]="wildjailbreak"
    #["sentiment"]="amazonreviews yelp twitterfinance twittermteb auditorsentiment newsmtc imdb financial_phrasebank dair_emotion sst5"
    #["topic"]="nytimes bbcnews agnews"
    #["fv"]="healthver climatefever fever"
    #["toxicity"]="unintended_jigsaw jigsaw"
    ["confidence"]="naturalqa msmarco triviaqa"
    ["bullets"]="naturalqa msmarco triviaqa"
    #["json"]="naturalqa msmarco triviaqa"
    ["verbconfidence"]="naturalqa msmarco triviaqa"
)
random_seeds=(42 609 101 32 55)
label_col="model_label"

source proj_params.sh
bash scripts/label_all.sh
for random_seed in "${random_seeds[@]}"
do
    model_save_name="${model_name#*/}"
    for task in "${!tasks_and_datasets[@]}"
    do
        datasets=${tasks_and_datasets[$task]}
        for dataset in $datasets
        do
            echo "XXXXXXXXXXXXXXX Running Ablations for $task $dataset $random_seed XXXXXXXXXXXXXXXXXXX"
            python ablations.py --task $task --dataset $dataset --random_seed $random_seed --label_col $label_col
        done
    done
done