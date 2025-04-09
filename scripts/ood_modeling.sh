tasks=("mcqa_2_ops" "sentiment")
model_name="meta-llama/Llama-3.1-8B-Instruct"
train_strategy="median"
model_kind="linear"
layer=18
label_col="model_label"

source proj_params.sh
#bash scripts/label_all.sh
model_save_name="${model_name#*/}"
for task in "${tasks[@]}"
do
    echo "XXXXXXXXXXXXXXX Running OOD for $task XXXXXXXXXXX"
    python ood_modeling.py --task $task --model_save_name $model_save_name --train_strategy $train_strategy --model_kind $model_kind --layer $layer --label_col $label_col
done