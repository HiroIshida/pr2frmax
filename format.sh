#!/bin/bash
find . -name "*py"|xargs python3 -m autoflake -i --remove-all-unused-imports --remove-unused-variables --ignore-init-module-imports
python3 -m isort ./ --profile black
python3 -m black --line-length 100 --target-version py38 --required-version 22.6.0 ./
python3 -m flake8 ./
