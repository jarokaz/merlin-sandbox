
```
docker build -t nvt-benchmark .
```

## SSD PD

```
docker run -it --rm --gpus all \
-v /mnt/disks/criteo:/data \
-v /home/jupyter/src:/src \
nvt-benchmark \
python /src/merlin-sandbox/nvt_benchmark/dask-nvtabular-criteo-benchmark.py \
--data-path /data/criteo_parquet \
--out-path /data/output_benchmark \
--devices "0,1" \
--device-limit-frac 0.8 \
--device-pool-frac 0.9 \
--num-io-threads 0 \
--part-mem-frac 0.125 \
--profile /data/output_benchmark/dask-report.html
```

## GCS using NVIDIA GCS connector

```
docker run -it --rm --gpus all \
-v /home/jupyter/src:/src \
nvt-benchmark \
python /src/merlin-sandbox/nvt_benchmark/dask-nvtabular-criteo-benchmark.py \
--data-path gs://jk-vertex-us-central1/criteo-parquet/criteo-parque \
--out-path gs://jk-vertex-us-central1/nvt_benchmark \
--devices "0,1" \
--device-limit-frac 0.8 \
--device-pool-frac 0.9 \
--num-io-threads 0 \
--part-mem-frac 0.125 \
--profile /src/dask-report.html
```

```
docker run -it --rm --gpus all \
-v /home/jupyter/src:/src \
nvt-benchmark \
python /src/merlin-sandbox/nvt_benchmark/dask-nvtabular-criteo-benchmark.py \
--data-path gs://workshop-datasets/criteo-parque \
--out-path gs://jk-vertex-us-central1/nvt_benchmark \
--devices "0,1" \
--device-limit-frac 0.8 \
--device-pool-frac 0.9 \
--num-io-threads 0 \
--part-mem-frac 0.125 \
--profile /src/dask-report.html
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
python train.py
```