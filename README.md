# STAMO

![](https://github.com/zhanglabtools/STAMO/blob/main/STAMO_overview.png)


## Overview

STAMO is designed for diagonal integration of unpaired spatial multi-omics data.

STAMO adopts a two-stages training strategy. In stage 1, pretraining graph attention network to produce coarse aligned embeddings. In stage 2, identifying anchors via Fused Gromov-Wasserstein optimal transport and performing anchor guided alignment. STAMO can be applied to integrate unpaired spatial slices with distinct omics feature space (including DNA, CUT&Tag, ATAC, RNA, and Protein) and artificially generate paired multi-omics data from profiled single-omics slices.



## Installation
The STAMO package is developed based on the Python libraries [bedtools](https://anaconda.org/bioconda/bedtools), [Scanpy](https://scanpy.readthedocs.io/en/stable/), [PyTorch](https://pytorch.org/) and [PyG](https://github.com/pyg-team/pytorch_geometric) (*PyTorch Geometric*) framework, and can be run on GPU (recommend) or CPU.

It's recommended to create a separate conda environment for running STAMO:

```
#create an environment called env_STAMO
conda create -n env_STAMO python=3.8

#activate your environment
conda activate env_STAMO
```

Please ensure the required packages: [POT](https://pythonot.github.io/), [bedtools](https://anaconda.org/bioconda/bedtools), [Scanpy](https://scanpy.readthedocs.io/en/stable/), [PyTorch](https://pytorch.org/) and [PyG](https://github.com/pyg-team/pytorch_geometric) have been installed in advance.

- For bedtools, make sure the version is not lower than v2.29.2. You can install it as follows:

  ```
  conda install -c bioconda bedtools==2.30.0
  ```

- You need to choose the appropriate dependency PyTorch and PyG for your own CUDA environment, and we successfully run STAMO under the following pytorch==1.13.1+cu116 and torch-geometric==2.3.0 with CUDA Version: 11.6. You can install it as follows:

  ```
  pip install torch_geometric
  
  pip install "https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp38-cp38-linux_x86_64.whl"
  ```

- Other packages can be found in requirement.txt:

  ```
  pip install -r requirement.txt
  ```



Finally, you can install STAMO as follows:

```
git clone https://github.com/zhanglabtools/STAMO.git
cd STAMO-main
python setup.py build
python setup.py install
```



## Tutorials

Step-by-step tutorials are included in the `Tutorial` folder to show how to use STAMO. 

- Tutorial 1: Two-omics integration on P22 mouse brain slices (RNA and H3K27ac)
- Tutorial 2: Five-omics integration on P22 mouse brain slices (RNA, ATAC, H3K27ac, H3K27me3, and H3K4me3)
- Tutorial 3: Integration of mouse embryo brain slices across different developmental stages
- Tutorial 4: Integration of slide-DNA-seq and slide-RNA-seq mouse liver metastasis slices
- Tutorial 5: Integration of spatial transcriptomics and proteomics human lymph node slices


## Data

The GTF file used for prior feature graph construction can be downloaded from GENCODE: ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz or [here](https://drive.google.com/file/d/1sHA15AAuwieKfukjuXcTKe4UrKVunRn3/view?usp=sharing).


## Support

If you have any questions, please feel free to contact us [xzhou@amss.ac.cn](mailto:xzhou@amss.ac.cn) or zhouxiang2@gdiist.cn. 




## Acknowledgement
This model borrows code for model training from [scGLUE](https://github.com/gao-lab/GLUE) and [STAGATE](https://github.com/zhanglabtools/STAGATE). We thank the respective authors for making their code available to the community.



## Citation

