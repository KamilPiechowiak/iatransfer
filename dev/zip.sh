#!/bin/bash

rm -rf iatransfer.zip && zip -q -r iatransfer.zip --exclude=.git** --exclude=venv/* --exclude=.idea/* --exclude=__pycache__/* --exclude=*.egg-info/* --exclude=stats/* .
