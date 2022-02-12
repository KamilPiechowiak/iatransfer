#!/bin/bash

rsync -r --exclude=__pycache__ iatransfer config scripts/put requirements.txt requirements_research.txt inf136780@polluks.cs.put.poznan.pl:iatransfer