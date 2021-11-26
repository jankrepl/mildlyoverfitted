OUTPUT_FOLDER=log_dir

python trainer.py --max-iter 1000 linear $OUTPUT_FOLDER/linear
python trainer.py --max-iter 1000 --shuffle-on-reset linear $OUTPUT_FOLDER/linear_augment
python trainer.py --max-iter 1000 MLP $OUTPUT_FOLDER/MLP
python trainer.py --max-iter 2000 --shuffle-on-reset MLP $OUTPUT_FOLDER/MLP_augment
python trainer.py --max-iter 14000 invariant $OUTPUT_FOLDER/invariant
