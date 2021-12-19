# Parallel parameters
N_JOBS=4
ARGS="-P$N_JOBS --header :" # arguments for parallel
# ARGS="--bar "$ARGS
ARGS="--dry-run "$ARGS

# Experiment parameters
ENTITY='mildlyoverfitted'
PROJECT='lottery_parallel_2'  # it should already exist to avoid issues

MAX_ITERS=(15000)
PRUNE_ITERS=(1 5)
PRUNE_METHODS=('l1' 'random')
PRUNE_RATIOS=(0 0.1 0.25 0.5 0.8 0.9 0.93 0.97)
REINITIALIZES=('true' 'false')
RANDOM_STATES=(1 2 3 4 5)

parallel $ARGS \
    python main.py \
        --max-iter={max_iter} \
        --prune-iter={prune_iter} \
        --prune-method={prune_method} \
        --prune-ratio={prune_ratio} \
        --random-state={random_state} \
        --reinitialize={reinitialize} \
        --wandb-entity=$ENTITY \
        --wandb-project=$PROJECT \
            ::: max_iter "${MAX_ITERS[@]}" \
            ::: prune_iter "${PRUNE_ITERS[@]}" \
            ::: prune_method "${PRUNE_METHODS[@]}" \
            ::: prune_ratio "${PRUNE_RATIOS[@]}" \
            ::: random_state "${RANDOM_STATES[@]}" \
            ::: reinitialize "${REINITIALIZES[@]}" \
