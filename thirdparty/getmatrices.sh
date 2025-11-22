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

if [ ! -d "mycielskian19" ]; then
  echo "Downloading mycielskian19..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/Mycielski/mycielskian19.tar.gz
  tar -xzf mycielskian19.tar.gz
  rm mycielskian19.tar.gz
  ls mycielskian19/mycielskian19.mtx
else
  echo "Circuit5M already exists"
fi

if [ ! -d "dielFilterV3clx" ]; then
  echo "Downloading kron_g500-logn21..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/Dziekonski/dielFilterV3clx.tar.gz
  tar -xzf dielFilterV3clx.tar.gz
  rm dielFilterV3clx.tar.gz
  ls dielFilterV3clx/dielFilterV3clx.mtx
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

if [ ! -d "msdoor" ]; then
  echo "Downloading audikw_1..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/INPRO/msdoor.tar.gz
  tar -xzf msdoor.tar.gz
  rm msdoor.tar.gz
  ls msdoor/msdoor.mtx
else
  echo "audikw_1 already exists"
fi

# volendo auto

