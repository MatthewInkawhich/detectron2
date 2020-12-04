#!/bin/bash

kill $(ps aux | grep "train_net.py" | grep -v grep | awk '{print $2}')
