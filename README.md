# PFM_MSHBM_MHVerbGen

**Author: Han Wang (2016)**

This respiratory contains the codes for precision functional mapping (PFM; [Gorden et al., 2017](https://www.cell.com/neuron/fulltext/S0896-6273(17)30613-X)) and functional connectivity (FC) analyses for data from [Krishnan et al. (2021)](https://www.sciencedirect.com/science/article/pii/S1053811920310843). The PFM is based on a multi-session hierarchical Bayesian model (MS-HBM) pipeline developed by [Kong et al. (2019)](https://academic.oup.com/cercor/article/29/6/2533/5033556?login=false). The current pipeline generates brain network parcellation on a single participant whose resting-state data has been pre-processed using the [Human Connectome Project (HCP) pipeline](https://github.com/Washington-University/HCPpipelines) with a denoising procedure using the [ICA-FIX](https://fsl.fmrib.ox.ac.uk/fsl/docs/resting_state/fix.html) pipeline.

The whole-brain FC analysis is based on the parcellation defined by the Cole-Anticevic Brain Network Atlas (CAB-NP; [Ji et al., 2019](https://www.sciencedirect.com/science/article/abs/pii/S1053811918319657); see [GitHub repo](https://github.com/ColeLab/ColeAnticevicNetPartition) here). Region-of-interst (ROI) FC analysis used nodes defined on the functional cortical atlas from [Glasser et al. (2016)](https://www.nature.com/articles/nature18933), which was the basis of the cortical parcels in Ji et al. (2019).


**_./res0ources/_** - Scripts for helper functions and reading and writing the CIFTI data, using the [MSCcodebase](https://github.com/MidnightScanClub/MSCcodebase).

**_./run_subject_mshbm.m_** - Run a single-subject MS-HBM pipeline and output parcellation figures.

**_./1_run_subject_connectivity_extraction_cab-np.sh_** - Run a single subject whole-brain connectivity extraction based on the [CAB-NP atlas](https://github.com/ColeLab/ColeAnticevicNetPartition/blob/master/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii) using Connectome Workbench.

**_./2_run_subject_connectivity_analysis_cab-np.py_** - Run a single subject ROI connectivity analysis based on the [CAB-NP atlas](https://github.com/ColeLab/ColeAnticevicNetPartition/blob/master/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii), with cortical regions mapped from [Glasser's atlas](https://www.nature.com/articles/nature18933). ROIs contained parcels for anterior cingulate cortex, anterior insular, lateral prefrontal cortex, and basal ganglia (caudate and putamen).
