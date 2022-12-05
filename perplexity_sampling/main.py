import seqio
import argparse

import sentencepiece as spm


parser = argparse.ArgumentParser()
parser.add_argument("--build_tokenizer", default=False, type=bool)
parser.add_argument("--tokenizer_input", default=None, type=str)
parser.add_argument("--tokenizer_prefix", default=None, type=str)
parser.add_argument("--tokenizer_path", default=None, type=str)
args = parser.parse_args()

if args.build_tokenizer:
    spm.SentencePieceTrainer.train(
        input=args.tokenizer_input,
        model_prefix=args.tokenizer_prefix,
        model_type="word",
        vocab_size=1000000,
        character_coverage=1.0
        )

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

DEFAULT_SPM_PATH = args.tokenizer_path
OUTPUT_FEATURES = {
    "inputs": 
        seqio.Feature(
            vocabulary=seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH),
            add_eos=True,
            required=False),
    "targets":
        seqio.Feature(
            vocabulary=seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH),
            add_eos=True)
    }

TaskRegistry.add(
    "calculate_perplexity",
    # source=seqio.TfdsDataSource(
    #     tfds_name="super_glue/wsc.fixed:1.0.2", splits=["train"]),
    preprocessors=[
        functools.partial(preprocessors.wsc_simple, correct_referent_only=True),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[],
    output_features=OUTPUT_FEATURES)

def PP(w):

    def kneser_ney()

    return 10**()

def gaussian_sampling(W):

    exponent = (-1/beta) * ((PP(W) - X)/X)**2
    return alpha * e ** exponent


dataset = seqio.get_mixture_or_task("mix1").get_dataset(
    sequence_length={"inputs": 256, "targets": 128},
    split="train",
    shuffle=True,
    num_epochs=1,
    shard_info=seqio.ShardInfo(index=0, num_shards=10),
    use_cached=False,
    seed=42
)

# Print the first 5 examples.
for _, ex in zip(range(5), dataset.as_numpy_iterator()):
  print(ex)

# def calculate_perplexity(
#     dataset,
#     sequence_length,
#     output_features
#     ):
#     """MLM Objective"""
#     ds = dataset
#     ds = t5.data.preprocessors.select_random_chunk(
#         ds,
#         output_features=output_features,
#         feature_key='targets',
#         max_length=65536)
#     ds = t5.data.preprocessors.reduce_concat_tokens(
#         ds,
#         feature_key='targets',
#         batch_size=128)
#     ds = t5.data.preprocessors.split_tokens(
#         ds,
#         feature_key='targets',
#         min_tokens_per_segment=None,
#         max_tokens_per_segment=sequence_length['targets']
#         )
#     ds = t5.data.preprocessors.denoise(
#         ds,
#         output_features,
#         inputs_fn=noise_token_to_mask_token,
#         targets_fn=None,
#         noise_density=0.15,
#         noise_mask_fn=preprocessors.iid_noise_mask
#     )
#     return ds