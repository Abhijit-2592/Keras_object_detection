# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:13:18 2017

@author: Abhijit

Handy script to monitor utilization of GPU and CPU while training. Works only for NVIDIA GPU.

Usage: python resource_utilization -i 1
# The above runs the script indefinetly and refreshes it every 1 secs
"""
from __future__ import print_function
import time
import os
import argparse
from argparse import RawTextHelpFormatter
import datetime
import psutil

parser = argparse.ArgumentParser(description="Utility script to Monitor NVIDIA GPU and CPU usage",formatter_class=RawTextHelpFormatter)
parser.add_argument("-i","--interval",help="the time interval in seconds to monitor (default 1)",type=int,metavar='',default=int(1))
parser.add_argument("-mt","--max_time",help="the max time in minutes to run the program (default None(runs indefinetly))",metavar='',default=None)

args=parser.parse_args()
interval = int(args.interval)
if args.max_time:
    max_time = int(args.max_time)
else:
    max_time = None

if not max_time:
    check_time = time.time()
    while True:
        time.sleep(interval)
        os.system("clear")
        os.system("nvidia-smi --query-gpu=gpu_name,utilization.gpu,temperature.gpu,utilization.memory --format=csv")
        ep_time = time.time() - check_time
        cpu_usage = float(psutil.cpu_percent())
        cpu_memory = psutil.virtual_memory()[2] 
        print("\n CPU % Utillization, % Memory: {}, {}".format(cpu_usage,cpu_memory))
        print("Elapsed time : {}".format(str(datetime.timedelta(seconds=ep_time))))
else:
    check_time = time.time()
    start_time = time.time()
    while (time.time() - start_time) < max_time * 60:
        time.sleep(interval)
        os.system("clear")
        os.system("nvidia-smi --query-gpu=gpu_name,utilization.gpu,temperature.gpu,utilization.memory --format=csv")
        ep_time = time.time() - check_time
        cpu_usage = float(psutil.cpu_percent())
        cpu_memory = psutil.virtual_memory()[2] 
        print("\n CPU % Utillization, % Memory: {}, {}".format(cpu_usage,cpu_memory))
        print("Elapsed time : {}".format(str(datetime.timedelta(seconds=ep_time))))
    
    
        
