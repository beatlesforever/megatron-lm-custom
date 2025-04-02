import pynvml
import time
import json
import argparse
import os
import signal
import sys
import threading

gpu_data = {'timestamps': [], 'tx': [], 'rx': [], 'utilization': []}
write_threshold = 1000  # 定义每次写入文件的阈值
lock = threading.Lock()  # 用于线程间同步的锁

def signal_handler(sig, frame):
    global gpu_data
    print('Signal received:', sig)
    save_data()
    sys.exit(0)

def save_data():
    global gpu_data
    args = parse_arguments()
    with lock:
        with open(args.filename, 'a') as json_file:
            for ts, tx, rx, util in zip(gpu_data['timestamps'], gpu_data['tx'], gpu_data['rx'], gpu_data['utilization']):
                data = {'timestamp': ts, 'tx': tx, 'rx': rx, 'utilization': util}
                json.dump(data, json_file)
                json_file.write('\n')
        gpu_data = {'timestamps': [], 'tx': [], 'rx': [], 'utilization': []}
    print(f"PCIe utilization and GPU utilization data have been saved to {args.filename}")

def get_gpu_utilization():
    utilization = []
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        utilization.append(util.gpu)
    return utilization

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

def monitor_utilization():
    global gpu_data
    try:
        while True:
            current_time = time.time()
            utils = get_pcie_utilization()
            utilization = get_gpu_utilization()
            with lock:
                gpu_data['timestamps'].append(current_time)
                gpu_data['tx'].append(utils['tx'])
                gpu_data['rx'].append(utils['rx'])
                gpu_data['utilization'].append(utilization)
                if len(gpu_data['timestamps']) >= write_threshold:
                    save_data()
            time.sleep(0.01)
    except KeyboardInterrupt:
        save_data()
        print(f"PCIe utilization and GPU utilization data have been saved to {args.filename}")
        pynvml.nvmlShutdown()
        sys.exit(0)

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
    gpu_data = {'timestamps': [], 'tx': [], 'rx': [], 'utilization': []}
    with open(args.filename, 'w') as json_file:
        json.dump(gpu_data, json_file)

    monitor_thread = threading.Thread(target=monitor_utilization)
    monitor_thread.daemon = True  # 设置守护线程
    monitor_thread.start()
    monitor_thread.join()

if __name__ == "__main__":
    main()
