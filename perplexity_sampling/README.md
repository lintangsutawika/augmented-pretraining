

```python
%run perplexity_sampling/main.py \
    --build_matrix True \
    --task calculate_perplexity_c4 \
    --vocab_path c4.model \
    --ngram 5 \
    --checkpoint_prefix checkpoint_c4_5gram \
    --seq_length 256

```

If you have this error message
```
FailedPreconditionError: Error executing an HTTP request: libcurl code 77 meaning 'Problem with the SSL CA cert (path? access rights?)', error details: error setting certificate verify locations:  CAfile: /etc/ssl/certs/ca-certificates.crt CApath: none
	 when reading metadata of gs://t5-data/vocabs/cc_en.32000/sentencepiece.model
```

run this and start over

```
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export TENSORSTORE_CA_BUNDLE="/etc/ssl/certs/ca-bundle.crt"
```