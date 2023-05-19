# Singing Voice Conversion

Attention!!! Great update for this project in 2023-05-14 !!!

We successfully implement a New Framework of applying diffusion model in SVC task. The new framework is called DiffSVC++ V2, as shown in the following figure 

![framework](https://github.com/SLPcourse/MDS6002-222041023-BingliangLi/blob/main/fig1.jpg).  (a) The workflow of the SVC system. (b) The structure of our proposed framework DiffSVC++ V2. The initial MCEP predictor produces the deterministic candidate, and then the diffusion denoiser network refines it. (c) The structure of the classical diffusion model. It samples the MCEP directly from noise.

 Our results display a significant improvement in the quality of acoustic feature generation and training efficiency compared to the classical diffusion models framework!
 ![framework](https://github.com/SLPcourse/MDS6002-222041023-BingliangLi/blob/main/tab1.jpg)

To know more about SVC, you can read [this tutorial](https://www.zhangxueyao.com/data/SVC/tutorial.html).

## Dataset

We adopt two public datasets, Opencpop [1] and M4Singer [2], to conduct **many-to-one** singing voice conversion. Specifically, we consider [Opencpop](https://wenet.org.cn/opencpop/) (which is a single singer dataset) as target singer and use [M4Singer](https://github.com/M4Singer/M4Singer) (which is a 20-singer dataset) as source singers.

## DiffSVC++

We can utilize the following two stages to conduct **any-to-one** conversion:

1. **Acoustics Mapping Training** (Training Stage): This stage is to train the initial predictor and the diffusion model which can generate MCEP conditioned on F0 feature and PPG feature. We used L2 loss to train (more detail can be seen in latent_diffusion/config/latent_diffusion).

2. **Inference and Conversion** (Conversion Stage): Given any source singer's audio, firstly, extract its content features including F0, AP, textual content features, as well as the initial prediction result. Then, use the model of training stage to infer the converted acoustic features (MCEP). Finally, given F0, AP, and the converted SP, we utilize WORLD as vocoder to synthesis the converted audio.


### Requirements for feature extraction and WORLD-based SVC

```bash
## If you need CUDA support, add "--extra-index-url https://download.pytorch.org/whl/cu117" in this following

#for feature extraction
conda create -n ldm python=3.8.5
pip install torch==1.13.1 torchaudio==0.13.1
pip install pyworld==0.3.2
pip install diffsptk==0.5.0
pip install tqdm
pip install openai-whisper
pip install tensorboard
pip install librosa


```

### Requirements for latent diffusion

```bash

cd latent_diffusion
conda activate ldm
conda env update -f environment.yaml



```



### Dataset Preprocess

After you download the datasets, you need to modify the path configuration in `config.py`:

```python
# To avoid ambiguity, you are supposed to use absolute path.

# Please configure the path of your downloaded datasets
dataset2path = {"Opencpop": "[Your Opencpop path]",
    "M4Singer": "[Your M4Singer path]"}

# Please configure the root path to save your data and model
root_path = "[Root path for saving data and model]"
```

#### Opencpop

Transform the original Opencpop transcriptions to JSON format:

```bash
cd preprocess
python process_opencpop.py
```

#### M4Singer

Select some utterance samples that will be converted. We randomly sample 5 utterances for every singer:

```bash
cd preprocess
python process_m4singer.py
```

#### Extract f0 features (input)

To extract f0 features, we use librosa.pyin to extract f0 feature, they will be

```bash
cd preprocess
python extract_f0.py
```

For hyparameters, we use 44100 Hz sampling rate, 10 ms frame shift and 25 ms window size.
Before feeding into the diffusion model, they will be discretize into 300 bins and then embedding into a feature space.

#### Extract: Whisper Features (input)

To extract whisper features, we utilze the pretrained [multilingual models](https://github.com/openai/whisper#available-models-and-languages) of Whisper (specifically, `medium` model):

```bash
cd preprocess
python extract_whisper.py
```

*Note: If you face `out of memory` errors, open `preprocess/extract_whisper.py` and go to line 55. You could try to reduce the default `batch_size` of `extract_whisper_features(dataset, dataset_type, batch_size=30)`. You could also go to line 99 to use a more suitable model.*

### Acoustic Mapping Training (Training Stage)

#### Output: MCEP Features

To obtain MCEP features, first we adopt [PyWORLD](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) to extract spectrum envelope (SP) features, and then transform SP to MCEP by using [diffsptk](https://github.com/sp-nitech/diffsptk):

```bash
cd preprocess
python extract_mcep.py
```

For hyparameters, we use 44100 Hz sampling rate and 10 ms frame shift.

#### First Stage Predictor training and the first stage prediction

Firstly, we need to obtain a first stage predictor for MCEP. During training stage, we aim to train a mapping from textual content features to the target singer's acoustic features. In the implementation: (1) For output, we use 40d MCEP features as ground truth. (2) For condition, we utilize the last layer encoder's output of [Whisper](https://github.com/openai/whisper) as the content features (which is 1024d). (3) For acoustic model, we adopt a 6 layer Transformer.

##### training

```bash
cd model
python -u main.py --debug False --dataset 'Opencpop' --model 'Transformer' --lr 2.6e-5 --batch_size 8 --epochs 20
```

This will give up the trained first stage model and the prediction of the training and validation set.

##### infer

```bash
cd model
python -u converse.py \
--source_dataset 'M4Singer' --dataset_type 'test' \
--model_file '/mntnfs/med_data5/yiwenhu/diffsvc/Singing-Voice-Conversion-BingliangLi/model/ckpts/Opencpop/Transformer_lr_2.6e-05/18.pt' \
--target_singer_f0_file '/mntnfs/med_data5/yiwenhu/diffsvc/Singing-Voice-Conversion-BingliangLi/preprocess/M4Singer/F0/test_f0.pkl' \
--inference_dir '/mntnfs/med_data5/yiwenhu/diffsvc/Singing-Voice-Conversion-BingliangLi/model/ckpts/M4Singer/Transformer_eval_conversion/'
```

this will give you the first stage prediction of the test set of M4Singer dataset.

##### move the first stage prediction to the right place

```bash
cd preprocess
cp <your first_stage_predition_for_train> <your preprocess location>/Opencpop/first_stage/20/train.npy
cp <your first_stage_predition_for_val> <your preprocess location>/Opencpop/first_stage/20/test.npy
cp <your first_stage_predition_for_test> <your preprocess location>/M4Singer/first_stage/20/test.npy
```

#### Diffusion model Training

Before training! make sure you have already moved the preprocess data to the location of right place. (Especially for the first stage prediction!!!)

You may need to adjust the config in latent diffusion, which lies in latent_diffusion/configs/latent-diffusion/diffsvc_MCEP_x0.yaml

for line 72 is the location of the preprocess data
- data_path: /mntnfs/med_data5/yiwenhu/diffsvc/Singing-Voice-Conversion-BingliangLi/preprocess/


Then you can begin training~

```bash
cd latent_diffusion
conda activate ldm
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/<config_spec>.yaml -t
```

Best result may obtain in about 20 epochs.

The training result and best model will be shown in the folder of latent_diffusion/logs/<your config> 

### Inference and Conversion (Conversion Stage)

#### Conversion of the first stage model

```bash
cd model
sh run_conversion.sh
```

Then we can obtain the converted audio of the first stage model.

#### Conversion of the diffusion model

for the conversion of test

```bash
cd latent_diffusion
conda activate ldm
python infer.py --outdir outputs/diffsvc_MCEP_x0 --dataset M4Singer --checkpoint logs/diffsvc_MCEP_x0/checkpoints/best.ckpt --config configs/infer/diffsvc_MCEP_x0.yaml --ratio 0.5
```

for the conversion of val

```bash
cd latent_diffusion
conda activate ldm
python infer.py --outdir outputs/diffsvc_MCEP_x0_val --steps 50 --dataset Opencpop --checkpoint logs/diffsvc_MCEP_x0/checkpoints/best.ckpt --config configs/infer/diffsvc_MCEP_x0.yaml --ratio 0.5
```

#### evaluation

If you use the GPU from SRIBD and use sbatch to run infer, you can move the log file to evaluation folder, then run the metric.ipynb to get the evaluation result.