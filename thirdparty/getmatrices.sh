#!/bin/bash

if [ ! -d "1138_bus" ]; then
  wget https://suitesparse-collection-website.herokuapp.com/MM/HB/1138_bus.tar.gz
  gzip -d 1138_bus.tar.gz
  tar -xf 1138_bus.tar
  ls 1138_bus/1138_bus.mtx
fi


if [ ! -d "F1" ]; then
  wget https://suitesparse-collection-website.herokuapp.com/MM/Koutsovasilis/F1.tar.gz
  gzip -d F1.tar.gz
  tar -xf F1.tar
  ls F1/F1.mtx
fi

if [ ! -d "fl2010" ]; then
  wget https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/fl2010.tar.gzip
  gzip -d fl2010.tar.gz
  tar -xf fl2010.tar
  ls fl2010/fl2010.mxt

if [ ! -d "circuit5M" ]; then
wget https://suitesparse-collection-website.herokuapp.com/MM/Freescale/circuit5M.tar.gz
gzip -d circuit5M.tar.gz
tar -xf circuit5M.tar
ls circuit5M/circuit5M.mtx
fi
