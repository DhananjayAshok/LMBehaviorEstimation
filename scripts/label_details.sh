source proj_params.sh
model_name="meta-llama/Llama-3.1-8B-Instruct"
model_save_name="${model_name#*/}"
task="unanswerable"
datasets=("selfaware")
for dataset in "${datasets[@]}"
do
    echo XXXXXXX Doing $task $dataset XXXXXXX
    split="train"
    filepath="$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/detailed/$task/${dataset}_${split}_inference.csv"
    python label.py --filename $filepath --config "unanswerable"
    python label.py --filename $filepath --config "label_correct" --label_col "model_correct"
    split="test"
    filepath="$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/detailed/$task/${dataset}_${split}_inference.csv"
    python label.py --filename $filepath --config "unanswerable"
    python label.py --filename $filepath --config "label_correct" --label_col "model_correct"
done

task="mcqa_2_ops"
datasets=("cosmoqa")
for dataset in "${datasets[@]}"
do
    echo XXXXXXX Doing $task $dataset XXXXXXX
    split="train"
    filepath="$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/detailed/$task/${dataset}_${split}_inference.csv"
    python label.py --filename $filepath --config "mcqa_answer"
    python label.py --filename $filepath --config "mcqa_correct" --label_col "model_correct"
    split="test"
    filepath="$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/detailed/$task/${dataset}_${split}_inference.csv"
    python label.py --filename $filepath --config "mcqa_answer"
    python label.py --filename $filepath --config "mcqa_correct" --label_col "model_correct"
done

task="sentiment"
datasets=("twitterfinance")
for dataset in "${datasets[@]}"
do
    echo XXXXXXX Doing $task $dataset XXXXXXX
    split="train"
    filepath="$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/detailed/$task/${dataset}_${split}_inference.csv"
    python label.py --filename $filepath --config "sentiment"
    python label.py --filename $filepath --config "label_correct" --label_col "model_correct"
    split="test"
    filepath="$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/detailed/$task/${dataset}_${split}_inference.csv"
    python label.py --filename $filepath --config "sentiment"
    python label.py --filename $filepath --config "label_correct" --label_col "model_correct"
done

task="topic"
datasets=("nytimes")
for dataset in "${datasets[@]}"
do
    echo XXXXXXX Doing $task $dataset XXXXXXX
    split="train"
    filepath="$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/detailed/$task/${dataset}_${split}_inference.csv"
    python label.py --filename $filepath --config "topic_$dataset"
    split="test"
    filepath="$BEHAVIOR_ANTICIPATION_RESULTS_DIR/$model_save_name/detailed/$task/${dataset}_${split}_inference.csv"
    python label.py --filename $filepath --config "topic_$dataset"
done
