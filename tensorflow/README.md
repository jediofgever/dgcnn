# Dynamic Graph CNN for Learning on Point Clouds (TensorFlow)

## Point Cloud Classification
* Run the training script:

``` bash
python train.py
```

* Run the evaluation script after training finished:

``` bash
python3 batch_inference.py --model_path log1/epoch_80.ckpt --dump_dir log1/dump --output_filelist log1/output_filelist.txt --room_data_filelist meta/area1_data_label.txt --visu
```
