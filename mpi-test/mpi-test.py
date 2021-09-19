import argparse
import json
import logging
import os
import time




def main(args):
    
    print('********************')
    print(os.environ)
    return

    for epoch in range(args.num_epochs):
        logging.info(f'Starting epoch: {epoch}')
        time.sleep(args.delay)
        logging.info(f'Epoch {epoch} completed')



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--train_data',
                        type=str,
                        required=False,
                        default='/criteo_data/output/train/_file_list.txt',
                        help='Path to training data _file_list.txt')

    parser.add_argument('--num_epochs',
                        type=int,
                        required=False,
                        default=10,
                        help='Number of training epochs')
    
    parser.add_argument('--delay',
                        type=int,
                        required=False,
                        default=3,
                        help='Sleep time')
 
    args = parser.parse_args()

    return args  

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')

    args = parse_args()

    logging.info(f"Args: {args}")
    start_time = time.time()
    logging.info("Starting training")

    main(args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Training completed. Elapsed time: {}".format(elapsed_time))

