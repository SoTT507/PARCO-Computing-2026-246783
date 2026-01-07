#!/bin/bash

if [ ! -d "1138_bus" ]; then
  echo "Downloading bcsstk30..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/HB/1138_bus.tar.gz
  tar -xzf 1138_bus.tar.gz
  rm 1138_bus.tar.gz
  ls 1138_bus/1138_bus.mtx
else
  echo "1138_bus already exists"
fi
