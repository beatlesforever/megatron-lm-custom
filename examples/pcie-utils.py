import pynvml
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
    data['tx'].extend(gpu_data['tx'])
    data['rx'].extend(gpu_data['rx'])
    with open(args.filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"PCIe utilization data has been saved to {args.filename}")

def get_pcie_utilization():
    pcie_utils = {'tx': [], 'rx': []}
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        pcie_utils['tx'].append(pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES))
        pcie_utils['rx'].append(pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES))
    return pcie_utils

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
    pynvml.nvmlInit()
    gpu_data = {'tx': [], 'rx': []}
    with open(args.filename, 'w') as json_file:
        json.dump(gpu_data, json_file, indent=4)
    try:
        while True:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            utils = get_pcie_utilization()
            gpu_data['tx'].append([time.time(), utils['tx']])
            gpu_data['rx'].append([time.time(), utils['rx']])
            time.sleep(0.01)
            # if len(gpu_data['tx']) > 50000:
            #     with open(args.filename, 'r') as file:
            #         data = json.load(file)
            #     data['tx'].extend(gpu_data['tx'])
            #     data['rx'].extend(gpu_data['rx'])
            #     with open(args.filename, 'w') as json_file:
            #         json.dump(data, json_file, indent=4)
            #     gpu_data = {'tx': [], 'rx': []}
    except KeyboardInterrupt:
        with open(args.filename, 'r') as file:
            data = json.load(file)
        data['tx'].extend(gpu_data['tx'])
        data['rx'].extend(gpu_data['rx'])
        with open(args.filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"PCIe utilization data has been saved to {args.filename}")
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
