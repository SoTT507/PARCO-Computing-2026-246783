#!/bin/bash

if [ ! -d "bcsstk30" ]; then
  echo "Downloading bcsstk30..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk30.tar.gz
  tar -xzf bcsstk30.tar.gz
  rm bcsstk30.tar.gz
  ls bcsstk30/bcsstk30.mtx
else
  echo "bcsstk30 already exists"
fi

if [ ! -d "bcsstk36" ]; then
  echo "Downloading bcsstk36..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/Boeing/bcsstk36.tar.gz
  tar -xzf bcsstk36.tar.gz
  rm bcsstk36.tar.gz
  ls bcsstk36/bcsstk36.mtx
else
  echo "bcsstk36 already exists"
fi

if [ ! -d "rdb968" ]; then
  echo "Downloading rdb968..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/Bai/rdb968.tar.gz
  tar -xzf rdb968.tar.gz
  rm rdb968.tar.gz
  ls rdb968/rdb968.mtx
else
  echo "rdb968 already exists"
fi

if [ ! -d "bcsstk25" ]; then
  echo "Downloading bcsstk25..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk25.tar.gz
  tar -xzf bcsstk25.tar.gz
  rm bcsstk25.tar.gz
  ls bcsstk25/bcsstk25.mtx
else
  echo "bcsstk25 already exists"
fi

if [ ! -d "af23560" ]; then
  echo "Downloading af23560..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/Bai/af23560.tar.gz
  tar -xzf af23560.tar.gz
  rm af23560.tar.gz
  ls af23560/af23560.mtx
else
  echo "af23560"
fi

