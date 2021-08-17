
## Build container

```
docker build -t gcr.io/jk-mlops-dev/merlin-preprocess .
docker push gcr.io/jk-mlops-dev/merlin-preprocess
```


## Preprocess

```
docker run -it --rm --gpus all \
-v /mnt/disks/criteo:/data \
-v /home/jupyter/src/merlin-sandbox:/src \
-w /src \
nvcr.io/nvidia/merlin/merlin-training:0.5.3 \
python /src/hugectr/preprocess/preprocess_nvt.py \
--train_dir=/data/train_parquet \
--valid_dir=/data/valid_parquet \
--output_dir=/data/criteo_data \
--devices=0,1 \
--device_limit_frac=0.7 \
--device_pool_frac=0.8 \
--part_mem_frac=0.1 \
--num_io_threads=2 \
--freq_limit=6 \
--out_files_per_proc=8 \
--parquet_format=1 \
--criteo_mode=1 
```




## Train

```
docker build -t gcr.io/jk-mlops-dev/merlin-train .
docker push gcr.io/jk-mlops-dev/merlin-train
```


```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /mnt/disks/criteo:/data \
-v /home/jupyter/src/merlin-sandbox:/src \
nvcr.io/nvidia/merlin/merlin-training:0.6 \
python /src/hugectr/train/criteo_parquet.py
```

```
docker run -it --rm --gpus all \
-v /mnt/disks/criteo:/data \
-v /home/jupyter/src:/src \
nvcr.io/nvidia/merlin/merlin-training:0.6 \
python /src/merlin-sandbox/hugectr/train/train.py
```

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /mnt/disks/criteo:/data \
gcr.io/jk-mlops-dev/merlin-train \
python train.py \
--max_iter=5000 \
--eval_interval=500 \
--batchsize=2048 \
--train_data=/data/criteo_data/train/_file_list.txt \
--valid_data=/data/criteo_data/valid/_file_list.txt \
--gpus=0,1
```

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /mnt/disks/criteo:/data \
gcr.io/jk-mlops-dev/merlin-train \
python train.py \
--max_iter=5000 \
--eval_interval=500 \
--batchsize=2048 \
--train_data=gs://jk-vertex-us-central1/criteo_data/train/_file_list.txt \
--valid_data=gs://jk-vertex-us-central1/criteo_data/valid/_file_list.txt \
--gpus=0,1
```



## Create filestore

```
gcloud beta filestore instances create nfs-server \
--zone=us-central1-a \
--tier=BASIC_SDD \
--file-share=name="vol1",capacity=2TB \
--network=name="default"
```