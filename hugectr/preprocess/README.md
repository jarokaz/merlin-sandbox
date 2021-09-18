```
docker run -it --rm --gpus all \
-v /home/jupyter/merlin-sandbox/hugectr/preprocess:/src \
-v /home/jupyter/criteo_unprocessed:/criteo_data \
nvcr.io/nvidia/merlin/merlin-training:0.5.3 \
python /src/preprocess_nvt.py \
--train_folder /criteo_data/train \
--valid_folder /criteo_data/valid \
--output_folder /criteo_data/output \
--devices 0,1,2,3 \
--protocol tcp \
--device_limit_frac 0.8 \
--device_pool_frac 0.9 \
--num_io_threads 4 \
--part_mem_frac 0.08 \
--out_files_per_proc 8 \
--freq_limit 6 \
--shuffle PER_PARTITION



```


## Cardinalities

Based on training dataset day_0 - day_19

```
[16961592, 34319, 16768, 7378, 20132, 4, 6955, 1384, 63, 11492137, 914596, 289081, 11, 2209, 10737, 79, 4, 971, 15, 17618173, 5049264, 15182940, 364088, 12075, 102, 35]

```

Based on training dataset day_0 - day_2

```
[2839307, 28141, 15313, 7229, 19673, 4, 6558, 1297, 63, 2156343, 327548, 178478, 11, 2208, 9517, 73, 4, 957, 15, 2893928, 1166099, 2636476, 211349, 10776, 92, 35]

'2839307,28141,15313,7229,19673,4,6558,1297,63,2156343,327548,178478,11,2208,9517,73,4,957,15,2893928,1166099,2636476,211349,10776,92,35'
```