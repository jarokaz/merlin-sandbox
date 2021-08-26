# GKE Bandwidth and GCS tests


## Creating a cluster

```
gcloud beta container --project "jk-mlops-dev" clusters create "jk-gpu-cluster-1" \
--zone "us-central1-a" \
--no-enable-basic-auth \
--cluster-version "1.20.8-gke.900" \
--release-channel "regular" \
--machine-type "n1-standard-1" \
--image-type "COS_CONTAINERD" \
--disk-type "pd-standard" \
--disk-size "100" \
--metadata disable-legacy-endpoints=true \
--scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
--max-pods-per-node "110" \
--num-nodes "3" \
--logging=SYSTEM,WORKLOAD \
--monitoring=SYSTEM \
--enable-ip-alias\
--network "projects/jk-mlops-dev/global/networks/default" \
--subnetwork "projects/jk-mlops-dev/regions/us-central1/subnetworks/default" --no-enable-intra-node-visibility \
--default-max-pods-per-node "110" \
--no-enable-master-authorized-networks \
--addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
--enable-autoupgrade \
--enable-autorepair \
--max-surge-upgrade 1 \
--max-unavailable-upgrade 0 \
--enable-shielded-nodes \
--node-locations "us-central1-a" 

gcloud beta container --project "jk-mlops-dev" node-pools create "gpu-pool-1" \
 --cluster "jk-gpu-cluster-1" \
 --zone "us-central1-a" \
 --machine-type "n1-standard-96" \
 --accelerator "type=nvidia-tesla-t4,count=4" \
 --image-type "COS_CONTAINERD" \
 --disk-type "pd-standard" --disk-size "500" \
 --local-ssd-count "16" \
 --metadata disable-legacy-endpoints=true \
 --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
 --num-nodes "2" \
 --enable-autoupgrade \
 --enable-autorepair \
 --max-surge-upgrade 1 \
 --max-unavailable-upgrade 0 \
 --max-pods-per-node "110" \
 --node-locations "us-central1-a"
```