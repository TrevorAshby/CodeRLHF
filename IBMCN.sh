#!/bin/sh

curl 'https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz' > datasetIBM.tar.gz

tar xvfz datasetIBM.tar.gz Project_CodeNet/metadata

cat languages.txt | while read p ; do
  echo $p
  tar xvfz datasetIBM.tar.gz Project_CodeNet/data/*/$p/
done
