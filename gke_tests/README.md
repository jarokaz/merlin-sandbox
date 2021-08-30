# GKE Bandwidth and GCS tests


## Creating a cluster

```
gcloud beta container --project "jk-mlops-dev" clusters create "jk-gpu-cluster-1" \
--zone "us-west1-a" \
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
--num-nodes "1" \
--logging=SYSTEM,WORKLOAD \
--monitoring=SYSTEM \
--enable-ip-alias \
--network "projects/jk-mlops-dev/global/networks/default" \
--subnetwork "projects/jk-mlops-dev/regions/us-west1/subnetworks/default" --no-enable-intra-node-visibility \
--default-max-pods-per-node "110" \
--no-enable-master-authorized-networks \
--addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
--enable-autoupgrade \
--enable-autorepair \
--max-surge-upgrade 1 \
--max-unavailable-upgrade 0 \
--enable-shielded-nodes \
--node-locations "us-west1-a" 

gcloud beta container --project "jk-mlops-dev" node-pools create "gpu-pool-1" \
 --cluster "jk-gpu-cluster-1" \
 --zone "us-west1-a" \
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
 --node-locations "us-west1-a" \
 --enable-gvnic
```

```
gcloud container node-pools create gvnic-pool \
    --cluster=jk-gpu-cluster-1 \
    --accelerator type=nvidia-tesla-t4,count=4 \
    --machine-type n1-standard-96 \
    --num-nodes 2 \
    --zone us-west1-a \
    --enable-gvnic
```

```
kubectl run my-shell --rm -i --tty --image ubuntu -- bash
```


pod 1 IP : 10.80.1.6

pod 2 IP : 10.80.2.4

```
kubectl exec --stdin --tty gpu-pod-1 -- /bin/bash
kubectl exec --stdin --tty gpu-pod-2 -- /bin/bash
```

```
gcloud container clusters create jk-gpu-cluster-1 \
    --zone "us-west1-a" \
    --accelerator type=nvidia-tesla-t4,count=4 \
    --machine-type=n1-standard-96 \
    --enable-gvnic
```

```
gsutil perfdiag -t rthru -s 2048M -n 2*N -c N gs://jk-perfdiag				
```
