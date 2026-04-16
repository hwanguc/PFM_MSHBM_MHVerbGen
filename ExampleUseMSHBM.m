% define paths 
Paths{1} = '~/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen/res0urces/read_write_cifti'; % this is the path to location containing ft_read/write functions
Paths{2} = '~/Apps/Utils/CBIG-master'; % this is the path the Thomas Yeo's functions, that have been modified by us

addpath(genpath(Paths{1}));
addpath(genpath(Paths{2}));

load('MSHBM-Priors.mat');

% Define HCP subject info
Subject = '100307';
BaseDir = '~/Documents/Data/ucl/gos_ich/hcp_example_data';

% Load surfaces
MidthickSurfs{1} = fullfile(BaseDir, Subject, 'MNINonLinear/fsaverage_LR32k', [Subject '.L.midthickness.32k_fs_LR.surf.gii']);
MidthickSurfs{2} = fullfile(BaseDir, Subject, 'MNINonLinear/fsaverage_LR32k', [Subject '.R.midthickness.32k_fs_LR.surf.gii']);

% Load dtseries
C = ft_read_cifti_mod(fullfile(BaseDir, Subject, 'MNINonLinear/Results/rfMRI_REST1_RL', 'rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii'));
%C = ft_read_cifti_mod(fullfile(BaseDir, Subject, 'MNINonLinear/Results/rfMRI_REST1_RL', 'rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii'));


%load([Subdir '/func/rest/ConcatenatedCiftis/FD.mat']);
C.data = single(C.data); % remove high motion volumes, convert to single type;

% Output directory
OutDir = fullfile(BaseDir, Subject, 'mshbm_output');
mkdir(OutDir);

PriorWeight = 1; % this controls how much weight the spatial priors impose
Smoothness = 10; % controls how likely neighboring vertices are to belong to the same network
pfm_mshbm(C, MidthickSurfs, OutDir, PriorWeight, Smoothness, Params, Paths);