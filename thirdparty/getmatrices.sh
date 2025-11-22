#!/bin/bash

if [ ! -d "F1" ]; then
  echo "Downloading F1..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/Koutsovasilis/F1.tar.gz
  tar -xzf F1.tar.gz
  rm F1.tar.gz
  ls F1/F1.mtx
else
  echo "F1 already exists"
fi

if [ ! -d "Circuit5M" ]; then
  echo "Downloading Circuit5M..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/Circuit5M.tar.gz
  tar -xzf Circuit5M.tar.gz
  rm Circuit5M.tar.gz
  ls Circuit5M/Circuit5M.mtx
else
  echo "Circuit5M already exists"
fi

if [ ! -d "kron_g500-logn21" ]; then
  echo "Downloading kron_g500-logn21..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn21.tar.gz
  tar -xzf kron_g500-logn21.tar.gz
  rm kron_g500-logn21.tar.gz
  ls kron_g500-logn21/kron_g500-logn21.mtx
else
  echo "kron_g500-logn21 already exists"
fi

if [ ! -d "kron_g500-logn19" ]; then
  echo "Downloading kron_g500-logn19..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn19.tar.gz
  tar -xzf kron_g500-logn19.tar.gz
  rm kron_g500-logn19.tar.gz
  ls kron_g500-logn19/kron_g500-logn19.mtx
else
  echo "kron_g500-logn19 already exists"
fi

if [ ! -d "audikw_1" ]; then
  echo "Downloading audikw_1..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/Pothen/audikw_1.tar.gz
  tar -xzf audikw_1.tar.gz
  rm audikw_1.tar.gz
  ls audikw_1/audikw_1.mtx
else
  echo "audikw_1 already exists"
fi

