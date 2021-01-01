#!/bin/bash

rm -rf iatransfer.zip && zip -r iatransfer.zip --exclude=.git** --exclude=venv/* --exclude=.idea/* --exclude=__pycache__/* --exclude=*.egg-info/* .
