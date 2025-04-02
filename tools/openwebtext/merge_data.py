import glob
import sys
import json
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".",
        help="path where all the json files are located")

    parser.add_argument("--output_file", type=str, default="merged_output.json",
        help="filename where the merged json should go")

    args = parser.parse_args()

    data_path = args.data_path
    out_file = args.output_file

    data_files = glob.glob(data_path + '/*.txt')

    counter = 0
    
    # for fname in data_files:
    #     counter += 1

    #     if counter % 1024 == 0:
    #         print("Merging at ", counter, flush=True)

    #     json_objects = []
    #     with open(fname, 'r') as infile:
    #         lines = infile.readlines()
    #         for line in lines:
    #             json_objects.append({"text": line})
        
    #     with open(out_file, 'w') as json_file:
    #         for obj in json_objects:
    #             json_file.write(json.dumps(obj))
                
    with open(out_file, 'w') as outfile:
        for fname in data_files:
            counter += 1

            if counter % 1024 == 0:
                print("Merging at ", counter, flush=True)

            with open(fname, 'r') as infile:
                lines = infile.readlines()
                for line in lines:
                    outfile.write(json.dumps({"text":line})+'\n')


    print("Merged file", out_file, flush=True)
