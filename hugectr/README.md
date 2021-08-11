

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