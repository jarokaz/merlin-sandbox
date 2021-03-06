
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

```
docker run -it --rm --gpus all \
-v /mnt/disks/criteo:/data \
gcr.io/jk-mlops-dev/merlin-preprocess \
python preprocess.py \
--input_data_dir /data/criteo_parquet \
--output_dir /data/output_8_files \
--n_train_days 20 \
--n_val_days 4 \
--device_limit_frac 0.7 \
--device_pool_frac 0.7 \
--part_mem_frac 0.1 \
--out_files_per_proc 4
```

```
docker run -it --rm --gpus all \
gcr.io/jk-mlops-dev/merlin-preprocess \
python preprocess.py \
--input_data_dir gs://jk-vertex-us-central1/criteo-parquet/criteo-parque \
--output_dir gs://jk-vertex-us-central1/nvt-testing/output_test \
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
nvcr.io/nvidia/merlin/merlin-training:0.6 \
python /src/merlin-sandbox/hugectr/train/train.py
```

```
docker run -it --rm --gpus all \
-v /mnt/nfs:/data \
gcr.io/jk-mlops-dev/merlin-train \
python train.py \
--max_iter=5000 \
--eval_interval=500 \
--input_train=/data/criteo-processed/train \
--input_val=/data/criteo-processed/valid \
--num_gpus=0,1
```


## Create filestore

```
gcloud beta filestore instances create nfs-server \
--zone=us-central1-a \
--tier=BASIC_SDD \
--file-share=name="vol1",capacity=2TB \
--network=name="default"
```