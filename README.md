# METAFormer
Repository for my Master Thesis entitled "Towards Interpretable Brain Biomarker Extraction using Deep Learning for fMRI Prediction" and the accompanying MICCAI 2023 MLCN paper "Pretraining is all you need: A Multi-Atlas Transformer Framework for Autism Spectrum Disorder Classification" 

<<<<<<< HEAD
:pushpin: METAFormer pre-print is available at [arxiv](https://arxiv.org/abs/2307.01759)

![METAFormer](assets/metaformer_arch.png)

:pushpin: **FULL CODE RELEASE UPON PUBLICATION**
=======
:pushpin: METAFormer paper is available at [arxiv](https://arxiv.org/abs/2307.01759)

![METAFormer](assets/metaformer_arch.png)

---------
## Quickstart

Clone the repository:

```
git clone https://github.com/Lugges991/METAFormer
```

install the necessary dependencies:
```
pip install -r requirements.txt
```

To Download ABIDE I data you need the phenotypic data file which is available [here](http://www.nitrc.org/frs/downloadlink.php/4912), then run:
```
python3 download.py pheno_file.csv out_dir_cc200 --roi cc200
python3 download.py pheno_file.csv out_dir_aal --roi aal
python3 download.py pheno_file.csv out_dir_dos160 --roi dos160
```

Generate functional connectomes:
```
python3 connectome.py --path path_to_1D_files --output out_dir_aal
python3 connectome.py --path path_to_1D_files --output out_dir_cc200
python3 connectome.py --path path_to_1D_files --output out_dir_dos160
```

Create csv:
```
python3 gen_csv.py aal_dir cc200_dir dos160_dir --pheno_file pheno_file --output fc.csv
```

Run CV-pretraining-finetuning:
```
python3 main.py
```

Generate feature attributions and calculate mean max-sensitivity and infidelity for each (this might take some time):
```
python3 attribute.py
```

## Cite
If you use METAFormer in your research, please cite our paper:
```
@misc{mahler2023pretraining,
      title={Pretraining is All You Need: A Multi-Atlas Enhanced Transformer Framework for Autism Spectrum Disorder Classification}, 
      author={Lucas Mahler and Qi Wang and Julius Steiglechner and Florian Birk and Samuel Heczko and Klaus Scheffler and Gabriele Lohmann},
      year={2023},
      eprint={2307.01759},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
>>>>>>> fd798b9 (added code for release)
