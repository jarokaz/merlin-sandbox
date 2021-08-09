# merlin-sandbox


## Create an instance with local SSD


```
gcloud compute instances create jk-ssd \
   --project jk-mlops-dev \
   --zone us-central1-a \
   --machine-type n1-standard-32 \
   --accelerator type=nvidia-tesla-t4,count=2 \
   --maintenance-policy TERMINATE --restart-on-failure \
   --image-family common-cu110 \
   --image-project deeplearning-platform-release \
   --boot-disk-size 200GB \
   --metadata "install-nvidia-driver=True,proxy-mode=project_editors" \
   --scopes https://www.googleapis.com/auth/cloud-platform \
   --local-ssd interface=nvme \
   --local-ssd interface=nvme \
   --local-ssd interface=nvme \
   --local-ssd interface=nvme 
   
  gcloud notebooks instances register jk-ssd --location us-central1-a
```


```
docker run -it --rm --gpus all \
-v /mnt/disks/pdssd/training_data:/training_data \
-v /mnt/disks/pdssd/validation_data:/validation_data \
-v /mnt/disks/pdssd/output:/output \
gcr.io/jk-mlops-dev/merlin-preprocess \
python preprocess.py \
--training_data /training_data \
--validation_data /validation_data \
--output_path /output \
--device_limit_frac 0.6 \
--device_pool_frac 0.6 \
--part_mem_frac 0.08
```

```
docker run -it --rm --gpus all \
-v /home/jupyter/scaling_criteo/training_data:/training_data \
-v /home/jupyter/scaling_criteo/validation_data:/validation_data \
-v /home/jupyter/scaling_criteo/output:/output \
gcr.io/jk-mlops-dev/merlin-preprocess \
python preprocess.py \
--training_data /training_data \
--validation_data /validation_data \
--output_path /output \
--device_limit_frac 0.6 \
--device_pool_frac 0.6 \
--part_mem_frac 0.08
```
