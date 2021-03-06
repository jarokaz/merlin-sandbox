# Testing GCS performance with NVTabular

## Provision a test VM

From Cloud Shell

```
export PROJECT_ID=jk-mlops-dev
export INSTANCE_NAME="merlin-dev"
export VM_IMAGE_PROJECT="deeplearning-platform-release"
export VM_IMAGE_FAMILY="common-cu110"
export MACHINE_TYPE="a2-highgpu-2g"
export BUCKET_REGION="us-central1"
export BUCKET_NAME=gs://jk-criteo-bucket
export LOCATION="us-central1-c"
export ACCELERATOR_TYPE=NVIDIA_TESLA_A100
export ACCELERATOR_COUNT=2
export BOOT_DISK_SIZE=500



gcloud notebooks instances create $INSTANCE_NAME \
--location=$LOCATION \
--vm-image-project=$VM_IMAGE_PROJECT \
--vm-image-family=$VM_IMAGE_FAMILY \
--machine-type=$MACHINE_TYPE \
--accelerator-type=$ACCELERATOR_TYPE \
--accelerator-core-count=$ACCELERATOR_COUNT \
--boot-disk-size=$BOOT_DISK_SIZE \
--install-gpu-driver

```

After your instance has been created connect to JupyterLab and open a JupyterLab terminal.


## Prepare Criteo data

### Create a GCS bucket in the same region as your notebook instance

```
gsutil mb -l $BUCKET_REGION $BUCKET_NAME
```

### Copy Criteo parquet files
```
gsutil -m cp -r gs://workshop-datasets/criteo-parque $BUCKET_NAME/

```



## Run a benchmark

### Clone the repo
```
cd 
git clone https://github.com/merlin-on-vertex
cd merlin-on-vertex/nvtabular_benchmark

```

### Build a container

```
docker build -t nvt-test .
```

```
docker run -it --rm --gpus all \
-v /tmp:/out \
nvt-test \
python dask-nvtabular-criteo-benchmark.py \
--out-path gs://jk-criteo-bucket/nvt-tests/output  \
--devices "0,1,2,3" \
--device-limit-frac 0.8 \
--device-pool-frac 0.9 \
--num-io-threads 0 \
--part-mem-frac 0.12 \
--profile /out/dask-report.html \
--data-path gs://jk-criteo-bucket/criteo_16_per_file 
```

```
docker run -it --rm --gpus all \
-v /tmp:/out \
-v /home/jupyter/criteo:/data \
nvt-test \
python dask-nvtabular-criteo-benchmark.py \
--out-path /out/output  \
--devices "0,1,2,3" \
--device-limit-frac 0.8 \
--device-pool-frac 0.9 \
--num-io-threads 0 \
--part-mem-frac 0.12 \
--profile /out/dask-report.html \
--data-path /data/criteo_16_per_file
```

```
docker run -it --rm --gpus all \
-v /tmp:/out \
-v /home/jupyter/criteo:/data \
gcr.io/jk-mlops-dev/nvt-test \
python dask-nvtabular-criteo-benchmark.py \
--out-path /out/output  \
--devices "0,1,2,3" \
--device-limit-frac 0.8 \
--device-pool-frac 0.9 \
--num-io-threads 0 \
--part-mem-frac 0.12 \
--profile /out/dask-report.html \
--data-path /data/criteo_16_per_file
```

```
python dask-nvtabular-criteo-benchmark.py \
--out-path gs://jk-criteo-bucket/nvt-tests/output \
--devices "0,1,2,3" \
--device-limit-frac 0.8 \
--device-pool-frac 0.9 \
--num-io-threads 0 \
--part-mem-frac 0.12 \
--profile /home/jupyter/output/dask-report.html \
--data-path gs://jk-criteo-bucket/criteo_16_per_file 
```

```
python dask-nvtabular-criteo-benchmark.py \
--out-path /tmp/output \
--devices "0,1,2,3" \
--device-limit-frac 0.8 \
--device-pool-frac 0.9 \
--num-io-threads 0 \
--part-mem-frac 0.12 \
--profile /tmp/output/dask-report.html \
--data-path /home/jupyter/criteo/criteo_16_per_file 
```


```
docker build -t gcr.io/jk-mlops-dev/nvt-test .
docker push gcr.io/jk-mlops-dev/nvt-test
```