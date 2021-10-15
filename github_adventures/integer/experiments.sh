set -x

OUTPUT_PATH=results
GLOVE_PATH=glove.6B.300d.txt
SEQUENCES_PATH=raw_data.pkl
MAX_VALUE_EVAL=500

python glove.py --max-value-eval $MAX_VALUE_EVAL $GLOVE_PATH $OUTPUT_PATH/glove
python bert.py --max-value-eval $MAX_VALUE_EVAL $OUTPUT_PATH/BERT
python lstm.py \
    $SEQUENCES_PATH \
    $OUTPUT_PATH/LSTM \
    --batch-size 128 \
    --device cuda \
    --embedding-dim 128 \
    --hidden-dim 256 \
    --max-value-eval $MAX_VALUE_EVAL \
    --max-value 20000 \
    --n-epochs 20000 \
    --sequence-len 100
