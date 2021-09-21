# Copyright (c) 2021 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Standard Libraries
import argparse
import logging
import pprint
import time

from google.cloud import aiplatform


TEST_FILE = 'mpi-test.py'


def run(args):

    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=args.gcs_bucket
    )

    job_name = 'MPI_TESTING_{}'.format(time.strftime("%Y%m%d_%H%M%S"))

    if args.accelerator_num > 0: 
        machine_spec = {
            "machine_type": args.machine_type,
            "accelerator_type": args.accelerator_type,
            "accelerator_count": args.accelerator_num,
        }
    else:
        machine_spec ={
            "machine_type": args.machine_type,
        }

    worker_pool_specs =  [
        {
            "machine_spec": machine_spec,
            "replica_count": 1,
            "container_spec": {
                "image_uri": args.train_image,
                #"command": ["python", TEST_FILE],
                "args": [
                    'python',
                    'mpi_test.py' 
                ],
            },
        },
        {
            "machine_spec": machine_spec,
            "replica_count": 1,
            "container_spec": {
                "image_uri": args.train_image,
                #"command": ["python", TEST_FILE],
                #"args": [
                #    #'--train-data=' + args.train_data,
                #    "echo", "testing" 
                #],
            },
        }
    ]

    logging.info(f'Starting job: {job_name}')

    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
        
    )
    job.run(sync=True,
            restart_job_on_worker_restart=False,
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--project',
                        type=str,
                        default='jk-mlops-dev',
                        help='Project ID')
    parser.add_argument('--region',
                        type=str,
                        default='us-central1',
                        help='Region')
    parser.add_argument('--gcs_bucket',
                        type=str,
                        default='gs://jk-vertex-us-central1',
                        help='GCS bucket')
    parser.add_argument('--vertex_sa',
                        type=str,
                        default='training-sa@jk-mlops-dev.iam.gserviceaccount.com',
                        help='Vertex SA')
    parser.add_argument('--machine_type',
                        type=str,
                        #default='a2-highgpu-2g',
                        default='n1-standard-4',
                        help='Machine type')
    parser.add_argument('--accelerator_type',
                        type=str,
                        default='NVIDIA_TESLA_A100',
                        help='Accelerator type')
    parser.add_argument('--accelerator_num',
                        type=int,
                        #default=2,
                        default=0,
                        help='Num of GPUs')
    parser.add_argument('--train-image',
                        type=str,
                        default='gcr.io/jk-mlops-dev/vertex-hugectr-test',
                        help='Training image name')

    parser.add_argument('--train_data',
                        type=str,
                        default='/gcs/jk-vertex-us-central1/criteo-benchmark-test',
                        help='Criteo parquet data location')
    

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')

    logging.info(f"Args: {args}")

    run(args)