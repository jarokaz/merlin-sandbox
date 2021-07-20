

# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import json
import os
import logging
import time


def get_args():
    """Defines and parse commandline arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path",
        default="/tmp",
        type=str,
    )
    
    parser.add_argument(
        "--input_path",
        default="/tmp",
        type=str,
    )

    return parser.parse_args()

def main():
    args = get_args()
    

    logging.info('****Entering****')

    print(os.listdir(args.input_path))
    
    logging.info('**** Exiting ****')
    
    
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
