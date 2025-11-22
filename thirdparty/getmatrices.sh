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

if [ ! -d "af_shell7" ]; then
  echo "Downloading af_shell7..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell7.tar.gz
  tar -xzf af_shell7.tar.gz
  rm af_shell7.tar.gz
  ls af_shell7/af_shell7.mtx
else
  echo "af_shell7 already exists"
fi

if [ ! -d "mario002" ]; then
  echo "Downloading kron_g500-logn21..."
  wget https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/mario002.tar.gz
  tar -xzf mario002.tar.gz
  rm mario002.tar.gz
  ls mario002/mario002.mtx
else
  echo "dielFilterV3clx already exists"
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
  echo "msdoor"
fi

# volendo auto

