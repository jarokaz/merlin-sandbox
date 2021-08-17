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


TRAIN_FILE = 'train.py'


def run(args):

    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=args.gcs_bucket
    )

    worker_pool_specs =  [
        {
            "machine_spec": {
                "machine_type": args.machine_type,
                "accelerator_type": args.accelerator_type,
                "accelerator_count": args.accelerator_num,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": args.train_image,
                "command": ["python", "train.py"],
                "args": [
                    '--batchsize=' + str(args.batchsize),
                    '--train_data' + args.train_data, 
                    '--valid_data=' + args.valid_data,
                    '--slot_size_array=' + args.slot_size_array,
                    '--max_iter=' + str(args.max_iter),
                    '--eval_interval=' + str(args.eval_interval),
                    '--workspace_size_per_gpu=' + str(args.workspace_size_per_gpu),
                    '--gpus=' + ','.join(map(str, range(args.accelerator_num))),
                ],
            },
        }
    ]

    job_name = 'HUGECTR_{}'.format(time.strftime("%Y%m%d_%H%M%S"))

    logging.info(f'Starting job: {job_name}')

    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
    )
    job.run(sync=True)

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
                        default='a2-highgpu-2g',
                        help='Machine type')
    parser.add_argument('--accelerator_type',
                        type=str,
                        default='NVIDIA_TESLA_A100',
                        help='Accelerator type')
    parser.add_argument('--accelerator_num',
                        type=int,
                        default=2,
                        help='Num of GPUs')
    parser.add_argument('--train_image',
                        type=str,
                        default='gcr.io/jk-mlops-dev/merlin-train',
                        help='Training image name')

    parser.add_argument('--train_data',
                        type=str,
                        default='/gcs/jk-vertex-us-central1/criteo_data/train/_file_list.txt',
                        help='Training data location')
    parser.add_argument('--valid_data',
                        type=str,
                        default='/gcs/jk-vertex-us-central1/criteo_data/valid/_file_list.txt',
                        help='Validation data location')
    parser.add_argument('--max_iter',
                        type=int,
                        default=10000,
                        help='Num of training iterations')
    parser.add_argument('--eval_interval',
                        type=int,
                        default=1000,
                        help='Run evaluation after given number of iterations')
    parser.add_argument('--batchsize',
                        type=int,
                        default=2048,
                        help='Batch size')
    parser.add_argument('-s',
                        '--snapshot',
                        type=int,
                        required=False,
                        default=10000,
                        help='Saves a model snapshot after given number of iterations')
    parser.add_argument('--slot_size_array',
                        type=str,
                        required=False,
                        default='3673278,28922,15432,7229,19673,4,6605,1315,63,2733920,376536,191906,11,2209,9663,74,4,957,15,3752950,1427075,3390584,231060,10906,92,35',
                        help='Categorical variables cardinalities')
    parser.add_argument('--workspace_size_per_gpu',
                        type=int,
                        required=False,
                        default=1000,
                        help='Workspace size per gpu in MB')

    args = parser.parse_args()


    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')

    logging.info(f"Args: {args}")

    run(args)