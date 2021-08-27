#!/bin/bash

for lat in {0..80..20}
    do 
        month=9
        echo "Running at lat=$lat deg, in month=$month"
        python -W ignore /home/anqil/Documents/Python/Photochemistry/Pablo/Model_Pablo_v2.py $month $lat
    done
