# Laboratory of Computational Physics, PoD, Unipd, Data Repository

Welcome to the data repository for Tuscano Alessio's Master Thesis course LACP-B. 
This repository hosts all the data used and generated during the course.

## Overview

The course focuses on using and experimenting on common and emergent tools for data analysis for scientific purposes, such as classification, clustering, and regression of physical datasets.

## Contents

The repository is organized as follows:

- **[exercises]**: exercise folder
- **[SBI_model1]**: folder Simulation Based Inference project for 'quartic potential' model
- **[SBI_model2]**: folder Simulation Based Inference project for 'red blood cell' model

Along with the exercises of the course there is a project with the use of SBI (Simulation Based Inference).
Very shortly we used this technique to infer the parameters of a simple quartic model solved also analytically, to check the performance of our approach; and a more advanced red blood cell model. 

The red blood cell model could not be solved analytically so this approach was needed to infer the parameters for the physical model itself.
We also used Numba along other techniques to fasten as much as possible the physical simulation, gaining performances comparable to C with a fraction of the time needed to develop a C simulation from the ground up.

## Usage

All the jupyter notebooks are written in python.

## Citation

If you use any data from this repository in your own research or projects, please cite this repository.

## License

This repository is licensed under the GNU GPLv3 license.

## Contact

If you have any questions, feedback, or issues regarding the data in this repository, please feel free to contact:

- Alessio Tuscano: alessio@tuscano.it

## Acknowledgments

All the exercises are group projects with the contribution of @chiatrama, @bevittoria, @belfagor123
