# 'Studio della superficie potenziale della molecola d’acqua tramite machine learning' Data Repository

Welcome to the data repository for Tuscano Alessio's Bachelor Thesis, titled 'Studio della superficie potenziale della molecola d’acqua tramite machine learning'. This repository hosts all the data used and generated during the research conducted for the thesis.

## Overview

The research focuses on developing a neural network capable of predicting the properties that can be calculated by a DFT driven simulation such as the one in 'pw.x' code from quantum-espresso.
For more details see the bachelor thesis document.

## Contents

The repository is organized as follows:

- **[main/model_graph_[scenario]]**: folders for predictions with various scenario
  - Under these folders one can find graphs for multiple neural networks predictions: every image is named with the primary hyperparameters that has been changed during the development of the final neural network, so it is easy to refer to the corresponding saved model.
- **[main/saved_model]**: folder where the developed trained models are saved in '.keras' format
  - These models use the same naming scheme that is used for their corresponding graphs and plots for ease of retrieval.
- **[main/saved_dataset]**: folder where all generated datasets are stored
  - all datasets are stored with less rigid naming. Be aware of that.
  - note: dataset

[Continue listing all the folders/directories and their contents.]

## Usage

[Provide instructions on how to use the data provided in this repository. Include any relevant information on data formats, structures, or preprocessing steps if necessary.]

## Citation

If you use any data from this repository in your own research or projects, please cite the following:

## License

This repository is licensed under the GNU GPLv3 license.

## Contact

If you have any questions, feedback, or issues regarding the data in this repository, please feel free to contact:

- [Your Name]: [Your Email Address]

## Acknowledgments

[Optional: Acknowledge any individuals, organizations, or funding sources that contributed to your research or the data collection process.]

