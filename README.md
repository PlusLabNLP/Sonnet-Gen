# Sonnet-Gen
NAACL 2022: Zero-shot Sonnet Generation with Discourse-level Planning and Aesthetics Features [https://aclanthology.org/2022.naacl-main.262/](https://aclanthology.org/2022.naacl-main.262/)

- Step 1 - Keyword generation
- Step 2 - Rhyme word generation
- Step 3 - Add simile and imagery
- Step 4 - Decoding

A few notes:
- Both step 1&2 are located in the keyword folder. 
- To directly use the pretrained model, run inference_bart_keywords_gen.ipynb and load the model from [https://huggingface.co/FigoMe/sonnet_keyword_gen](https://huggingface.co/FigoMe/sonnet_keyword_gen). Then at decoding time, load the pretrained model from [https://huggingface.co/FigoMe/news-gpt-neo-1.3B-keywords-line-by-line-reverse](https://huggingface.co/FigoMe/news-gpt-neo-1.3B-keywords-line-by-line-reverse)
- To train the keyword model yourself, run train-keywords-bart.ipynb (we shifted from T5 to bart)

## Citations
Please cite our paper if they are helpful to your work !
``` @inproceedings{tian-peng-2022-zero,
    title = "Zero-shot Sonnet Generation with Discourse-level Planning and Aesthetics Features",
    author = "Tian, Yufei  and
      Peng, Nanyun",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.262",
    doi = "10.18653/v1/2022.naacl-main.262",
    pages = "3587--3597",
    abstract = "Poetry generation, and creative language generation in general, usually suffers from the lack of large training data. In this paper, we present a novel framework to generate sonnets that does not require training on poems. We design a hierarchical framework which plans the poem sketch before decoding. Specifically, a content planning module is trained on non-poetic texts to obtain discourse-level coherence; then a rhyme module generates rhyme words and a polishing module introduces imagery and similes for aesthetics purposes. Finally, we design a constrained decoding algorithm to impose the meter-and-rhyme constraint of the generated sonnets. Automatic and human evaluation show that our multi-stage approach without training on poem corpora generates more coherent, poetic, and creative sonnets than several strong baselines.",
} ```

