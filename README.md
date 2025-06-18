Hi, 

## Installations

Install the requirements
```
pip install -r requirements.txt
```

and then 

```
python testing_utils/testing/setup_testing.py
```


After that, download weights from this drive: [drive](https://drive.google.com/drive/folders/1En2BI9H9LxqA5XIpNaMXhqhF8--XAKns)


## Dataset Preprocessing
then run `datasets.ipynb` notebook completely to setup data

## Training
then verify and update paths in lines 22, 41, and 45 in `training_utils/train_linear_classifier_selfcon.py` file, and then run

```
python training_utils/train_linear_classifier_selfcon.py
```
you can also adjust no. of epochs by adjusting For loop in line 121.

## Inference
After training is completed, you can run inference by running: 

```
    python inference_origin.py \
        --data-root . \
        --ann     annotations/testgen_images.txt \
        --ckpt    linear_imaginet_save_origin_calib/model_5.pt \
        --bs      64 \
        --out     results_origin_finetuning_10.csv
```
In the above command

`ckpt` is the model's path
`ann` is the file that will contain paths of datasets, and what we generated in `datasets.ipynb` notebook. 
`out` is the output file where models results for each image in dataset will be saved

## Results and Evaluation

To evaluate the training, run `eval.ipynb` notebook completly.
