set -x 
SEED=$RANDOM

python train.py \
	--batch-size 128 \
	--beta 0.01 \
	--eval-frequency 4000 \
	--device cuda \
	--lambda-p 0.2 \
	--n-elems 30 \
	--n-iter 1500000 \
	--n-hidden 128 \
	--n-nonzero 1 25 \
	results/experiment_b/$SEED
