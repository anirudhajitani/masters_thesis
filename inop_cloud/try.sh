#!/bin/bash

export LOAD=$1
echo $LOAD
count=0
#cpulimit -l 6 dd if=/dev/zero of=/dev/null & export PID=$!
yes > /dev/null & export PID=$!
cpulimit -l 4 -p $PID & export PID_2=$! 
sleep $LOAD
kill -9 $PID
kill -9 $PID_2
wait $PID
wait $PID_2
#sleep 2
