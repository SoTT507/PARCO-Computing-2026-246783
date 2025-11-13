#!/bin/bash

wget https://suitesparse-collection-website.herokuapp.com/MM/HB/1138_bus.tar.gz
gzip -d 1138_bus.tar.gz
tar -xf 1138_bus.tar
ls 1138_bus/1138_bus.mtx



wget https://suitesparse-collection-website.herokuapp.com/MM/Koutsovasilis/F1.tar.gz
gzip -d F1.tar.gz
tar -xf F1.tar
ls F1/F1.mtx
