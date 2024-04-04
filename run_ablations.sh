#!/bin/bash

# crawl enron gist msong audio sift uqv 
for dataset in $@ ; do
    for method in parlayivf parlayivf-no-material-join parlayivf-no-sorted-queries parlayivf-no-bitvector parlayivf-no-weight-classes; do
        python run.py --neurips23track filter --algorithm $method --dataset $dataset 
    done
    # wait # if using, add & to call
    # sudo chmod 777 -R results
    # python plot.py --neurips23track filter --dataset $dataset
done


