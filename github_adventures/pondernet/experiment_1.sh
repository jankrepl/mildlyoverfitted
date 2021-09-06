set -x 
SEED=$RANDOM
LAMBDAS=(0.1 0.3 0.5 0.7 0.9)

for lambda in ${LAMBDAS[@]}
do
	python train.py \
		--batch-size 128 \
		--beta 0.01 \
		--device cuda \
		--eval-frequency 4000 \
		--n-iter 100000 \
		--n-hidden 128 \
		--lambda-p $lambda \
		--n-elems 15 \
		results/experiment_a/$SEED/lambda_$lambda
done
