# eeTranslator: train locally - predict on GEE server-side

This repository contains classes which accept fitted scikit-learn objects and translate the predict/classify step to GEE server side functions. 
Requires to have a GEE account. Please see the example: examples/rtm_inversion.ipynb

Currently implemented:
- sklearn.neural_networks.MLPRegressor --> eeMLPRegressor
- sklearn.preprocessing.StandardScaler --> eeStandardScaler

Contributions are welcome.

Developed within the Open-Earth-Monitor Cyberinfrastructure project: [Link](https://earthmonitor.org/)
