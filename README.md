# PFM_MSHBM_MHVerbGen

This respiratory contains the MATLAB code for precision functional mapping ([Gorden et al., 2017](https://www.cell.com/neuron/fulltext/S0896-6273(17)30613-X)) based on a multi-session hierarchical Bayesian model (MS-HBM) pipeline developed by [Kong et al. (2019)](https://academic.oup.com/cercor/article/29/6/2533/5033556?login=false). The current pipeline generates brain network parcellation on a single participant whose resting-state data has been pre-processed using the [Human Connectome Project pipeline](https://github.com/Washington-University/HCPpipelines) with a denoising procedure using the [ICA-FIX](https://fsl.fmrib.ox.ac.uk/fsl/docs/resting_state/fix.html) pipeline. 

**_./res0ources/_** - Scripts for reading and writing the CIFTI data, using the [MSCcodebase](https://github.com/MidnightScanClub/MSCcodebase).

**_./ExampleUseMSHBM.m_** - Run the MS-HBM pipeline and output parcellation figures.

**_./pfm_mshbm.m_** - The helper function that performs MS-HBM.

**_./MSHBM-Priors.mat_** - The prior weights/parameters for the MSHBM.

**_./calculate_network_surface_area.m_** - Calculate the surface area per network (UNTESTED!).
