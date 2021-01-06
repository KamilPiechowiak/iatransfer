#!/bin/bash

mkdir -p stats
gsutil -m rsync -r -x "^.*\.pt$|^.*\.png$|^.*\.tar\.gz$" gs://weights-transfer/ stats