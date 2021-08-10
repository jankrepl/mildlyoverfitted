set -x

N_EPOCHS=100000
N_SAMPLES=1000
SEED=123
TBOARD_DIR=tb_results/$SEED

python train.py -r $SEED -n $N_EPOCHS -s $N_SAMPLES $TBOARD_DIR/no_regularization
python train.py -r $SEED -n $N_EPOCHS -s $N_SAMPLES $TBOARD_DIR/weight_decay --weight-decay 0.6
python train.py -r $SEED -n $N_EPOCHS -s $N_SAMPLES $TBOARD_DIR/dropout -p 0.2 
python train.py -r $SEED -n $N_EPOCHS -s $N_SAMPLES $TBOARD_DIR/mixup --mixup 
python train.py -r $SEED -n $N_EPOCHS -s $N_SAMPLES $TBOARD_DIR/input_mixup -k 0 1 --mixup
python train.py -r $SEED -n $N_EPOCHS -s $N_SAMPLES $TBOARD_DIR/hidden_layers_mixup -k 1 4 --mixup 
