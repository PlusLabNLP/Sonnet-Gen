# Sonnet-Gen
NAACL 2022: Zero-shot Sonnet Generation with Discourse-level Planning and Aesthetics Features [https://aclanthology.org/2022.naacl-main.262/](https://aclanthology.org/2022.naacl-main.262/)

- Step 1 - Keyword generation
- Step 2 - Rhyme word generation
- Step 3 - Add simile and imagery
- Step 4 - Decoding

Step 1&2 are located in the keyword folder. 
- To directly use the pretrained model, run inference_bart_keywords_gen.ipynb and load the model from [https://huggingface.co/FigoMe/sonnet_keyword_gen](https://huggingface.co/FigoMe/sonnet_keyword_gen).
- To train the keyword model yourself, run train-keywords-bart.ipynb (we shifted from T5 to bart)


