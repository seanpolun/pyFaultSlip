# pyFaultSlip
pyFaultSlip is a program to perform 2D and 3D fault slip tendency analysis using a probabilistic approach (e.g. Morris et al., 1996 *Geology* ; Walsh and Zoback, 2016, *Geology*). This software has been completed with the intention of providing a straightforward means of assessing induced slip hazards from deep fluid (e.g. CO2 or wastewater) for researchers, regulators, operators, and other stakeholders. Since this was developed to support a specific project, we welcome input from the community to make this a more inclusive tool. 

## Installation
This program is implemented wholly in python 3.7+. It is highly recommended to use the [conda](https://conda.io/en/latest/) package manager to install all the dependencies. 

pyFaultSlip depends on the following: 
* [numpy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [geopandas](https://geopandas.org/index.html)
* [shapely](https://github.com/Toblerity/Shapely)
* [meshio](https://github.com/nschloe/meshio)
* [trimesh](https://github.com/mikedh/trimesh)
* [numba](https://numba.pydata.org/)
* [pyqt5](https://pypi.org/project/PyQt5/)

  To create a conda environment:
```
  conda create -n pyfaultslip python=3.11
  conda activate pyfaultslip
  conda install -c conda-forge numpy matplotlib scipy geopandas shapely meshio trimesh numba pyqt
```
  

## Usage
pyFaultSlip works as both a command-line function / module and as a GUI interface (currently only implemented for 2D lineaments). 

Input settings and parameters are input as JSON files. FaultSlipMain.main() is a helper function to condition JSON data to input data classes. 

Documentation forthcoming. 

## Acknowledgements 
This software was developed with support in part by the U.S. Department of Energy grant number DE-FE0031623. Neither the U.S. Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights.

(c) 2020-2024 Sean G. Polun (MIT License)
