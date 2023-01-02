#!/bin/bash
for i in {102400..409600..10240}
do
    ./ex_2_nonstreamed $i    
done

for i in {102400..409600..10240}
do
    ./ex_2_streamed $i 4   
done