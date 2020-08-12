# Job_Description_104_Segmentation_by_CKIP

This is an example of how to apply [CKIPtagger](https://github.com/ckiplab/ckiptagger) to Pandas DataFrame.

## Requirements
CKIPtagger requires:
* python>=3.6
* tensorflow>=1.13.1,<2 / tensorflow-gpu>=1.13.1,<2 (one of them)
* to use gpu: Nvidia GPU driver [CUDA Toolkit 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) (for Tensorflow 1.13/1.15) and [cudNN v7.6.5](https://developer.nvidia.com/rdp/cudnn-archive) (for CUDA 10.0)
## Speeding Up
### GPU Usage
The lines that enable GPU usage was included but commented out. 
```
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
```
#ws = WS('data', disable_cuda=False)
#pos = POS("./data", disable_cuda=False)
#ner = NER("./data", disable_cuda=False)
```
### Multi-Processing / Multi-Threading
As CKIPtagger model is defaultly set to use multi-processsing to speed up the calculation, multi-threading (pool) is added here to maximize the CPU usage.
