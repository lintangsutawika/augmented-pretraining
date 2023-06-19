
python build_dataset.py \
    --lang "en" \
    --data_frame_path "/fsx/lintangsutawika/03-cross-lingual-knowledge/dump/enwiki/link_en.parquet.gzip" \
    --connection $1 \
    --save_path /fsx/lintangsutawika/03-cross-lingual-knowledge/dataset/ \
    --samples 10_000_000 \
    --file_suffix "_part_$(printf %02d ${SLURM_NODEID})" \
    --seed ${SLURM_NODEID}