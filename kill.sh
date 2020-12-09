#!/bin/bash

kill $(ps aux | grep "inkawhmj/itd/bin" | grep -v grep | awk '{print $2}')
