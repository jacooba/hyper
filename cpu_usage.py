import psutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--interval', type=int, default=10)
args = parser.parse_args()
print(psutil.cpu_percent(interval=args.interval, percpu=True))