## Replicating Merlin container crash

### Install Vertex SDK

```
pip install -U google-cloud-aiplatform
```

### Configure settings

```
PROJECT=merlin-on-gcp
REGION=us-central1
STAGING_BUCKET=gs://merlin-jobs-staging
```

### Build container

```
IMAGE_NAME=gcr.io/$PROJECT/merlin-test

docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME
```

### Submit a Vertex job

```

python run.py --project=$PROJECT --region=$REGION --gcs_bucket=$STAGING_BUCKET --train_image=$IMAGE_NAME
```

