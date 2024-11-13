# Improving CLIP Counting Accuracy via Parameter-Efficient Fine-Tuning

This is the code repository for paper **Improving CLIP Counting Accuracy via Parameter-Efficient Fine-Tuning** 


## Abstract
We focus on addressing the object counting limitations of vision-language models, with a particular emphasis on Contrastive Language-Image Pre-training (CLIP) models. Centered on our hypothesis that counting knowledge can be abstracted into linear vectors within the text embedding space, we develop a parameter-efficient fine-tuning method and several zero-shot methods to improve CLIP's counting accuracy. Through comprehensive experiments, we demonstrate that our learning-based method not only outperforms full-model fine-tuning in counting accuracy but also retains the broad capabilities of pre-trained CLIP models. Our zero-shot text embedding editing techniques are also effective in situations where training data is scarce, and can be extended to improve Stable Diffusion's ability to generate images with precise object counts.We also contribute two specialized datasets to train and evaluate CLIP’s counting capabilities.


## Reproduce experiment results
### Install required packaged
Create your own virtual environment using a python environment > 3.6
```
conda create -y -n ENV_NAME
conda activate ENC_NAME
cd $CODE_DIR
pip install -r requirements.txt
``` 
### Download dataset
[To be posted]

### CLIP experiments
Run `python run.py` with specified configurations:
*  `--model`: choose CLIP model from ["clip_base_32","clip_base_16","clip_large_14"];
*  `--dataset`: choose dataset from ["custom","countbench"];
* `data_path`: specify the path to custom data if you run on the `custom` dataset;
* `eval_dir`: directory to save evaluation results
* `--task`: choose task from ["classification","image_retrievel"];
* `--ref_obj`: specify the name of reference object in a string format;

### Stable Diffusion experiments
Run `python run.py` with specified configurations:
* `-m`, `--model`: set as "stable_diffusion";
* `-t`, `--task`: set as "image_gen";
* `-r`, `--ref_obj`: specify the name of reference object in a string format;
