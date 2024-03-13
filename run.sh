declare -A TEST_PATH_DICT
TEST_PATH_DICT=(
    [ACE]="./dataset_processing/ACE05_processed/test.json" \
    [MAVEN]="./dataset_processing/MAVEN/valid.jsonl" \
    [ERE]="./dataset_processing/ERE/test.jsonl"  \
)
export TEST_FILE=${TEST_PATH_DICT[$DATA]}
export TRAIN_FILE=./dataset_processing/k_shot/fewshot_set/K${K}_${DATA}_${idx}/train.pkl
export VALID_FILE=./dataset_processing/k_shot/fewshot_set/K${K}_${DATA}_${idx}/dev.pkl
export LABEL_DICT_PATH=./dataset_processing/k_shot/label_dict_${DATA}.json
export OUT_PATH=./outputs/$DATA/${K}-shot/${idx}
mkdir -p $OUT_PATH
echo $OUT_PATH

queue_size=8192
if [ "$K" -eq 2 ] && { [ "$DATA" == 'ACE' ] || [ "$DATA" == 'ERE' ]; }; then
    queue_size=2048
fi

python main.py \
    --output_dir $OUT_PATH \
    --train_file $TRAIN_FILE \
    --dev_file $VALID_FILE \
    --test_file $TEST_FILE \
    --label_dict_path $LABEL_DICT_PATH \
    --max_steps 200 \
    --batch_size 128 \
    --logging_steps 10 \
    --eval_steps 10 \
    --use_label_semantics \
    --use_normalize \
    --learning_rate 1e-4 \
    --dataset_type $DATA \
    --queue_size $queue_size \
    --start_eval_steps 50 \
    --max_seq_length 192 \
    --fp_16 \
    --drop_none_event 
