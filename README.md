# merlin-sandbox


## Create an instance with local SSD


```
gcloud compute instances create jk-ssd \
   --project jk-mlops-dev \
   --zone us-central1-a \
   --machine-type n1-standard-32 \
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
