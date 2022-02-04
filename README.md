# Detecting Polarized Topics Using Partisanship-aware Contextualized Topic Embeddings
This repo implements this [paper](https://aclanthology.org/2021.findings-emnlp.181.pdf).

## Installation
1. Download data.zip from [here](https://drive.google.com/file/d/1RlZ-w-XnpOi45mqZPsMHKFIB7tpjyzxt/view?usp=sharing) to 01_topic_modeling/ and unzip it.
2. Install [Pytorch](https://pytorch.org/get-started/locally/) and [Huggingface Transformers](https://huggingface.co/docs/transformers/installation).


## Usage
1. Data preprocessing and topic modeling. Go to 01_topic_modeling and run 01_data_loading.ipynb, 02_data_tokenization.ipynb, and 03_lda_topic_modeling.ipynb. The data has already been saved, so you can skip running these three notebooks. However, it is recommended that you go through them.


2. Corpus-contextualized embedding generation and topic polarization ranking. Set the parameters in <em>run.py</em> and run the following code
```
python run.py
```
After this, you will get the ranking of topics based on polarization in 02_cc_emb_gen/results.


By default, it will run the results on the nine partisan news pairs. If just want to run on one pair, you can put just one source in <em>source1</em> and <em>source2</em>.

By comparing the results of <em>gt</em> (ground truth) and <em>emb</em> (PaCTE) you can get the recall reported at Table 3 in the paper.

## Citation
```angular2html
@inproceedings{he2021detecting,
  title={Detecting Polarized Topics Using Partisanship-aware Contextualized Topic Embeddings},
  author={He, Zihao and Mokhberian, Negar and C{\^a}mara, Ant{\'o}nio and Abeliuk, Andres and Lerman, Kristina},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2021},
  pages={2102--2118},
  year={2021}
}
```