
## Preprocess

```
docker run -it --rm --gpus all \
-v /mnt/disks/criteo:/data \
gcr.io/jk-mlops-dev/merlin-preprocess \
python preprocess.py \
--input_data_dir /data/criteo_parquet \
--output_dir /data/output_test \
--n_train_days 2 \
--n_val_days 1 \
--device_limit_frac 0.7 \
--device_pool_frac 0.7 \
--part_mem_frac 0.1
```


## Train


```
docker run -it --rm --gpus all \
-v /mnt/disks/criteo:/data \
-v /home/jupyter/src:/src \
nvcr.io/nvidia/merlin/merlin-training:0.5.3 \
python /src/merlin-sandbox/hugectr/train/train.py
```

```
docker run -it --rm --gpus all \
-v /mnt/disks/criteo:/data \
gcr.io/jk-mlops-dev/merlin-train \
python train.py \
--max_iter=5000 \
--eval_interval=500
```