#!/bin/bash

kill $(ps aux | grep "inkawhmj/itd/bin" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "tester" | grep -v grep | awk '{print $2}')
