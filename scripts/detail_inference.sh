declare -A tasks_and_datasets=(
    ["mcqa_2_ops"]="cosmoqa"
    ["unanswerable"]="selfaware"
    ["sentiment"]="twitterfinance"
    ["topic"]="nytimes"
)
splits=("train" "test")
model_name="meta-llama/Llama-3.1-8B-Instruct"
#model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
#model_name="mistralai/Mistral-7B-Instruct-v0.3"



max_new_tokens=125
source proj_params.sh
model_save_name="${model_name#*/}"


for task in "${!tasks_and_datasets[@]}"
do
    datasets=${tasks_and_datasets[$task]}
    for dataset in $datasets
    do
        for split in "${splits[@]}"
        do
            echo "XXXXXXXXXXXXXXXX $task $dataset $split XXXXXXXXXXXXXXXX"
            data_path="$BEHAVIOR_ANTICIPATION_DATA_DIR/$task/${dataset}_$split.csv"
            output_csv_path="$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/detailed/$task/${dataset}_${split}_inference.csv"
            output_hidden_dir="$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/detailed/$task/$split/${dataset}"
            python detail_inference.py --model_name $model_name --data_path $data_path --output_csv_path $output_csv_path --output_hidden_dir $output_hidden_dir --max_new_tokens $max_new_tokens
        done
    done
done