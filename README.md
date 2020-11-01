# CPP
Code for the Cell Point Process (Bioarxiv link to come)

## Environment

CPP was implemented with pytorch 1.4.0 and numpy 1.18.1.

## Usage

See Example_Notebook.ipynb for an example of how to simulate data, create a video representation, and estimate parameters to the model.

## In this repo

Data used for this paper can be found at https://www.dropbox.com/sh/ctrb51chmkyfhlt/AABH1A1jBFrVSahljz7VSbWaa?dl=0

### frontier

* run_frontier.py: Python code that randomly generates parameters, simulates data across different numbers of cells and peaks, and then estimates parameters from that data
* send_frontier.sh: A bash script for sending multiple run_frontier.py jobs to the clusters. We simulated with numpy random seeds 1-50.
* frontier-figs-Copy1.ipynb: Jupyter notebook for producing the figures seen in the paper
* learned- .npz: The output from 50 simulations

### tapi

* run_tapi.py: Python code for loading data from tapi_dose.mat and estimating parameters for each well
* learned-103120.npz: The npz output of run_tapi.py
* figs_and_stats_Cop1.ipynb: Creates figures showing estimated paramters vs. tapi concentration
* lieklihood-test.ipynb: Creates graph of relative likelihood to a control model vs. tapi concentration
* spatial_lambda_ratio_plots: Creates graph of percent of conditional intensity vs tapi concentrations

### drug

* run_drug_screen.py: Python code for loading data from FINAL_EQ.mat (Goglia, 2019) and estimating parameters for each well
* sample-matching-edited.xlsx: Excel table of metadata for each well
* param-control-fixed- - .npz: Output of run_drug_screen.py for different segments of FINAL_EQ.mat
* drug-figures-clean.ipynb: Jupyter notebook to produce figures showing 3d plot of different parameters for each class

### gfp

* run_gfp.py: Python code for loading data from dataforcpp.mat and estimating parameters for each well
* learned-102920.npz: Output of run_gfp.py (all wells)
* gfp-figs-Copy1.ipynb: Jupyter notebook to view results. The GFP figure was created in biorender.

### wound healing
