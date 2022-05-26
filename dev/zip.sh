#!/bin/bash

rm -rf iatransfer.zip && zip -q -r -v iatransfer.zip --exclude=**__pycache__** --exclude=.git** --exclude=venv/* --exclude=.idea/* --exclude=*.egg-info/* --exclude=stats/* .
