% Paths to required function libraries
Paths{1} = '~/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen/res0urces/read_write_cifti';           % ft_read_cifti_mod, etc.
Paths{2} = '~/Apps/Utils/CBIG-master';          % CBIG + MS-HBM modified functions

addpath(genpath(Paths{1}));
addpath(genpath(Paths{2}));


% Load prior template
load('~/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen/MSHBM-Priors.mat');            % Contains Params.NetworkLabels + Params.NetworkColors

% Define HCP subject info
Subject = '100307';
BaseDir = '~/Documents/Data/ucl/gos_ich/hcp_example_data';

% Load surfaces
MidthickSurfs{1} = fullfile(BaseDir, Subject, 'MNINonLinear/fsaverage_LR32k', [Subject '.L.midthickness.32k_fs_LR.surf.gii']);
MidthickSurfs{2} = fullfile(BaseDir, Subject, 'MNINonLinear/fsaverage_LR32k', [Subject '.R.midthickness.32k_fs_LR.surf.gii']);

% Load dtseries
C = ft_read_cifti_mod(fullfile(BaseDir, Subject, 'MNINonLinear/Results/rfMRI_REST1_RL', 'rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii'));

% Optional: clean timepoints with FD

motion = load(fullfile(BaseDir, Subject, 'MNINonLinear/Results/rfMRI_REST1_RL', 'Movement_Regressors.txt'));

% Extract the 6 realignment parameters
% Columns: [X, Y, Z translations, pitch, yaw, roll]
trans = motion(:, 1:3);   % in mm
rot = motion(:, 4:6);     % in radians

% Convert rotations to mm assuming a 50mm head radius
rot_mm = rot * 50;

% Framewise displacement = sum of absolute differences
FD = [0; sum(abs(diff([trans rot_mm])), 2)];

C.data = single(C.data(:, FD < 0.3));

% Output directory
OutDir = fullfile(BaseDir, Subject, 'mshbm_output');
mkdir(OutDir);

% Run MSHBM
PriorWeight = 1;
Smoothness = 10;
pfm_mshbm(C, MidthickSurfs, OutDir, PriorWeight, Smoothness, Params, Paths);

% Locate .dlabel.nii output
%DlabelPath = fullfile(OutDir, ['MS-HBM_FunctionalNetworks_VertexWiseThresh0.01_w' num2str(PriorWeight) '_c' num2str(Smoothness) '.dlabel.nii']);

% Vertex area file
%VA = fullfile(BaseDir, Subject, 'MNINonLinear/fsaverage_LR32k', [Subject '.midthickness_va.32k_fs_LR.dscalar.nii']);

% Save network size to CSV
%OutCSV = fullfile(OutDir, 'network_surface_area.csv');
%calculate_network_surface_area(DlabelPath, VA, Params.NetworkLabels, OutCSV);