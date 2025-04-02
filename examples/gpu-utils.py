import subprocess
import time
import json
import argparse
import os
import signal
import sys

gpu_data = {'tx': [], 'rx': []}

def signal_handler(sig, frame):
    global gpu_data
    print('Signal received:', sig)
    save_data()
    sys.exit(0)

def save_data():
    global gpu_data
    args = parse_arguments()
    with open(args.filename, 'r') as file:
        data = json.load(file)
    data.extend(gpu_data)
    with open(args.filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"GPU utilization data has been saved to {args.filename}")


def get_gpu_utilization():
    cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
    try:
        output = subprocess.check_output(cmd, shell=True)
        gpu_utils = output.decode().strip().split('\n')
        gpu_utils = [int(x) for x in gpu_utils]  # 转换为整数列表
        return gpu_utils
    except subprocess.CalledProcessError as e:
        print(f"Error during subprocess call: {e}")
        return []

def parse_arguments():
    parser = argparse.ArgumentParser(description="Monitor GPU utilization and save to JSON file.")
    parser.add_argument("filename", type=str, help="The filename where the GPU utilization data will be stored.")
    return parser.parse_args()

def main():
    global gpu_data
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_arguments()
    index = args.filename.rfind('/')
    if index != -1:
        dirname = args.filename[:index+1]
    directory = os.path.dirname(dirname)
    if directory != '':
        os.makedirs(directory, exist_ok=True)
    gpu_data = []
    with open(args.filename, 'w') as json_file:
        json.dump(gpu_data, json_file, indent=4)
    try:
        while True:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            utils = get_gpu_utilization()
            # print(f"{current_time} - GPU Utilization: {utils}", len(gpu_data))
            # gpu_data.append({"time": current_time, "utilization": utils})
            gpu_data.append([time.time(), utils])
            time.sleep(0.005)
            # if len(gpu_data) > 50000:
            #     # print(f"{current_time} - GPU Utilization: {gpu_data}")
            #     with open(args.filename, 'r') as file:
            #         data = json.load(file)
            #     data.extend(gpu_data)
            #     with open(args.filename, 'w') as json_file:
            #         json.dump(data, json_file, indent=4)
            #     gpu_data = []
    except KeyboardInterrupt:
        with open(args.filename, 'r') as file:
            data = json.load(file)
        data.extend(gpu_data)
        with open(args.filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"GPU utilization data has been saved to {args.filename}")

if __name__ == "__main__":
    main()
