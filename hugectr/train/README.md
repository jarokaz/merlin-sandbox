
## CLI snippets 

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
-v /mnt/disks/criteo:/criteo_data \
gcr.io/jk-mlops-dev/merlin-train \
python train.py \
--max_iter=100000 \
--eval_interval=1000 \
--batchsize=2048 \
--train_data=/criteo_data/criteo_processed/train/_file_list.txt \
--valid_data=/criteo_data/criteo_processed/valid/_file_list.txt \
--workspace_size_per_gpu=2000 \
--display_interval=500 \
--gpus=0,1
```

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /mnt/disks/criteo:/criteo_data \
gcr.io/jk-mlops-dev/merlin-train \
python train.py \
--num_epochs 1 \
--max_iter 500000 \
--eval_interval=5000 \
--batchsize=4096 \
--snapshot=0 \
--train_data=/criteo_data/criteo_processed/train/_file_list.txt \
--valid_data=/criteo_data/criteo_processed/valid/_file_list.txt \
--workspace_size_per_gpu=9000 \
--display_interval=1000 \
--gpus=0,1
```

```
```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/criteo_processed:/criteo_processed \
gcr.io/jk-mlops-dev/merlin-train \
python train.py \
--num_epochs 1 \
--max_iter 500000 \
--eval_interval=5000 \
--batchsize=2048 \
--snapshot=0 \
--train_data=/criteo_processed/train/_file_list.txt \
--valid_data=/criteo_processed/valid/_file_list.txt \
--workspace_size_per_gpu=9000 \
--display_interval=1000 \
--gpus=0
```
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


# Train


```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-sandbox/hugectr/train:/src \
-v /home/jupyter/criteo_processed:/criteo_data \
nvcr.io/nvidia/merlin/merlin-training:0.6 \
python /src/dcn_parquet.py 
```



```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-sandbox/hugectr/train:/src \
-v /home/jupyter/criteo_processed:/criteo_data \
nvcr.io/nvidia/merlin/merlin-training:0.6 \
python /src/train.py \
--num_epochs 1 \
--max_iter 500000 \
--eval_interval=7000 \
--batchsize=4096 \
--snapshot=0 \
--train_data=/criteo_data/output/train/_file_list.txt  \
--valid_data=/criteo_data/output/valid/_file_list.txt  \
--display_interval=1000 \
--workspace_size_per_gpu=2000 \
--gpus=0,1
```

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-sandbox/hugectr/train:/src \
-v /home/jupyter/criteo_processed:/criteo_data \
-w /src \
nvcr.io/nvidia/merlin/merlin-training:0.6 \
python -m trainer.task \
--num_epochs 1 \
--max_iter 50000 \
--eval_interval=600 \
--batchsize=16384 \
--snapshot=0 \
--train_data=/criteo_data/output/train/_file_list.txt  \
--valid_data=/criteo_data/output/valid/_file_list.txt  \
--display_interval=200 \
--workspace_size_per_gpu=1 \
--slot_size_array="[2839307, 28141, 15313, 7229, 19673, 4, 6558, 1297, 63, 2156343, 327548, 178478, 11, 2208, 9517, 73, 4, 957, 15, 2893928, 1166099, 2636476, 211349, 10776, 92, 35]" \
--gpus="[[0,1]]"
```

```
docker run -it --rm --gpus all --cap-add SYS_NICE --network host \
-v /home/jupyter/criteo_processed:/criteo_data \
gcr.io/jk-mlops-dev/merlin-train \
python -m trainer.task \
--num_epochs 1 \
--max_iter 50000 \
--eval_interval=600 \
--batchsize=16384 \
--snapshot=0 \
--train_data=/criteo_data/output/train/_file_list.txt  \
--valid_data=/criteo_data/output/valid/_file_list.txt  \
--display_interval=200 \
--workspace_size_per_gpu=1 \
--slot_size_array="[2839307, 28141, 15313, 7229, 19673, 4, 6558, 1297, 63, 2156343, 327548, 178478, 11, 2208, 9517, 73, 4, 957, 15, 2893928, 1166099, 2636476, 211349, 10776, 92, 35]" \
--gpus="[[0,1]]"
```