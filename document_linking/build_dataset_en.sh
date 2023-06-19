
CONNECTION=$1

python build_dataset.py \
    --lang "id" \
    --data_frame_path "/fsx/lintangsutawika/03-cross-lingual-knowledge/dump/enwiki/link_en.parquet.gzip" \
    --connection ${CONNECTION} \
    --save_path /fsx/lintangsutawika/03-cross-lingual-knowledge/dataset/en/ \
    --samples 1000 \
    --file_suffix "_part_$(printf %02d ${SLURM_NODEID})" \
    --seed ${SLURM_NODEID}