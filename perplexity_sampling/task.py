"""
To cache tasks before training,
seqio_cache_tasks \
    --tasks=my_task_*,your_task \
    --excluded_tasks=my_task_5 \
    --output_cache_dir=/path/to/cache_dir \
    --module_import=my.tasks \
    --alsologtostderr
For more details, see: seqio/scripts/cache_tasks_main.py
"""

import os
import seqio
import functools

import tensorflow as tf

TaskRegistry = seqio.TaskRegistry

# @seqio.map_over_dataset
# def extract_text_from_jsonl_tf(json: str):
#     output = tf.strings.split(json, '{"text": "', maxsplit=1)[1]
#     output = tf.strings.split(output, '",', maxsplit=1)[0]
#     return {"text": output}

@seqio.map_over_dataset
def extract_text_from_json_tf(json: str):
    output = tf.strings.split(json, '{"text":"', maxsplit=1)[1]
    output = tf.strings.split(output, '",', maxsplit=1)[0]
    return {"text": output}

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"
OUTPUT_FEATURES = {
    "text":
        seqio.Feature(
            vocabulary=seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH),
            add_eos=True,
            required=False),
    }

TaskRegistry.add(
    "calculate_perplexity_mt5",
    source=seqio.TextLineDataSource(
        split_to_filepattern={
            "train": ["/fsx/lintangsutawika/c4-train.00974-of-01024.json"],
            }
        ),
        preprocessors=[
            extract_text_from_json_tf,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
    ],
    output_features=OUTPUT_FEATURES,
    metric_fns=[]
)
