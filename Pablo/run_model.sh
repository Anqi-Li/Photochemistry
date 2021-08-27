#!/bin/bash

for lat in {0..80..20}
    do 
    # month = 6
    for month in 7 8 10 11
        do
        echo "Running at lat=$lat deg, in month=$month"
        python -W ignore /home/anqil/Documents/Python/Photochemistry/Pablo/Model_Pablo_v2.py $month $lat
        done
    done
