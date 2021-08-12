

```
docker run -it --rm --gpus all \
-v /mnt/disks/criteo:/data \
-v /home/jupyter/src:/src \
nvcr.io/nvidia/merlin/merlin-training:0.6
```
```

docker run -it --rm --gpus all \
gcr.io/jk-mlops-dev/hugectr_train_test \
python train.py \
--input_train gs://jk-vertex-us-central1/criteo-processed/train \
--input_val gs://jk-vertex-us-centra1l/criteo-processed/valid
```


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