declare -A tasks_and_datasets=(
    ["mcqa_2_ops"]="mmlu cosmoqa piqa arc medmcqa commonsenseqa openbookqa qasc hellaswag bigbenchhard_mcq truthfulqa"
)
splits=("train" "test")
model_name="meta-llama/Llama-3.1-8B-Instruct"



declare -A max_new_token_dict=( ["mcqa_2_ops"]=125)
source proj_params.sh
model_save_name="${model_name#*/}"


for task in "${!tasks_and_datasets[@]}"
do
    max_new_tokens=${max_new_token_dict[$task]}
    datasets=${tasks_and_datasets[$task]}
    for dataset in $datasets
    do
        for split in "${splits[@]}"
        do
            echo "XXXXXXXXXXXXXXXX $task $dataset $split XXXXXXXXXXXXXXXX"
            data_path="$COTA_DATA_DIR/$task/${dataset}_$split.csv"
            output_csv_path="$COTA_RESULTS_DIR/$model_save_name/$task/${dataset}_${split}_inference.csv"
            output_hidden_dir="$COTA_RESULTS_DIR/$model_save_name/$task/$split/${dataset}"
            python save_inference.py --model_name $model_name --data_path $data_path --output_csv_path $output_csv_path --output_hidden_dir $output_hidden_dir --max_new_tokens $max_new_tokens --save_hidden False --output_column "fewshot_output"
        done
    done
done