
```
docker build -t gcr.io/jk-mlops-dev/nvt-benchmark .
docker push gcr.io/jk-mlops-dev/nvt-benchmark 
```

## Local PD

```
docker run -it --rm --gpus all \
-v /mnt/disks/criteo:/data \
gcr.io/jk-mlops-dev/nvt-benchmark \
python dask-nvtabular-criteo-benchmark.py \
--data-path /data/criteo-benchmark-test \
--out-path /data/nvt_benchmark \
--devices "0,1" \
--device-limit-frac 0.8 \
--device-pool-frac 0.9 \
--num-io-threads 0 \
--part-mem-frac 0.125 \
--profile /data/output_benchmark/dask-report.html
```

```
docker run -it --rm --gpus all \
-v /home/jupyter/src:/out \
gcr.io/jk-mlops-dev/nvt-benchmark \
python dask-nvtabular-criteo-benchmark.py \
--data-path gs://jk-vertex-us-central1/criteo-benchmark-test \
--out-path gs://jk-vertex-us-central1/nvt_benchmark \
--devices "0,1" \
--device-limit-frac 0.8 \
--device-pool-frac 0.9 \
--num-io-threads 0 \
--part-mem-frac 0.125 \
--profile  /out/dask-report.html
```





