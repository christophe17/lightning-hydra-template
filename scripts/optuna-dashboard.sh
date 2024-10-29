#!/bin/bash

docker run -it --rm -p 8080:8080 --platform linux/amd64 -v `pwd`:/app -w /app  ghcr.io/optuna/optuna-dashboard sqlite:///optuna.db