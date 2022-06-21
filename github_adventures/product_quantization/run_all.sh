set -ex

# Parameters
URL=https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
RAW_FASTTEXT=raw_fasttext.vec
MAX_WORDS=100000
OUTPUT_FOLDER=new_results  # no slash
SCIKIT_KWARGS='--n_init 1 --max_iter 30 --init random'

# Download fasttext embeddings
if [ ! -f $RAW_FASTTEXT ]
then
    curl $URL --output $RAW_FASTTEXT.gz
    gzip -d $RAW_FASTTEXT.gz
fi

mkdir $OUTPUT_FOLDER

# Parse raw data
python parse.py $RAW_FASTTEXT $OUTPUT_FOLDER -m $MAX_WORDS

# Generate a couple of different indexes
python generate_index.py \
    $OUTPUT_FOLDER/embs.npy \
    faiss-flat \
    $OUTPUT_FOLDER/flat.faiss

python generate_index.py \
    $OUTPUT_FOLDER/embs.npy \
    faiss-pq \
    $OUTPUT_FOLDER/faisspq_m4_nbits8.faiss \
    --m 4 \
    --nbits 8

python generate_index.py \
    $OUTPUT_FOLDER/embs.npy \
    faiss-pq \
    $OUTPUT_FOLDER/faisspq_m12_nbits8.faiss \
    --m 12 \
    --nbits 8

python generate_index.py \
    $OUTPUT_FOLDER/embs.npy \
    our-pq \
    $OUTPUT_FOLDER/custompq_m4_nbits8.pkl \
    --m 4 \
    --nbits 8 \
    $SCIKIT_KWARGS

python generate_index.py \
    $OUTPUT_FOLDER/embs.npy \
    our-pq \
    $OUTPUT_FOLDER/custompq_m12_nbits8.pkl \
    --m 12 \
    --nbits 8 \
    $SCIKIT_KWARGS

# Convert faiss index into custom index
python convert.py \
    $OUTPUT_FOLDER/faisspq_m12_nbits8.faiss \
    $OUTPUT_FOLDER/converted_faisspq_m12_nbits8.pkl


# Run webapp

GRADIO_SERVER_PORT=7777 python run_gradio.py \
    $OUTPUT_FOLDER/flat.faiss \
    $OUTPUT_FOLDER/faisspq_m12_nbits8.faiss \
    $OUTPUT_FOLDER/converted_faisspq_m12_nbits8.pkl \
    $OUTPUT_FOLDER/words.txt \
    &

GRADIO_SERVER_PORT=7778 python run_gradio.py \
    $OUTPUT_FOLDER/flat.faiss \
    $OUTPUT_FOLDER/faisspq_m4_nbits8.faiss \
    $OUTPUT_FOLDER/faisspq_m12_nbits8.faiss \
    $OUTPUT_FOLDER/words.txt \
    &


GRADIO_SERVER_PORT=7779 python run_gradio.py \
    $OUTPUT_FOLDER/flat.faiss \
    $OUTPUT_FOLDER/custompq_m4_nbits8.pkl \
    $OUTPUT_FOLDER/custompq_m12_nbits8.pkl \
    $OUTPUT_FOLDER/words.txt \
    &
# make sure to kill the gradio processes pkill -f gradio
