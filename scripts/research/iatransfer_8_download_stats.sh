#!/bin/bash

mkdir -p stats
gsutil -m rsync -r -x "^.*\.pt$|^.*\.tar\.gz$" gs://weights-transfer/ stats