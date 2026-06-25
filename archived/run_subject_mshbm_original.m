% Define paths 
Paths{1} = '~/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen/res0urces/read_write_cifti'; % this is the path to location containing ft_read/write functions
Paths{2} = '~/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen/res0urces/helper_functions'; % this is the path to other helper functions
Paths{3} = '~/Apps/Utils/CBIG-master'; % this is the path the Thomas Yeo's functions, that have been modified by us

addpath(genpath(Paths{1}));
addpath(genpath(Paths{2}));
addpath(genpath(Paths{3}));

load('MSHBM-Priors.mat');

% Define subject info (Krishnan et al. 2021, processed with adapted HCP pipeline)
Subject = 'sub-509BT';
BaseDir = '~/Documents/Data/ucl/gos_ich/verb_gen_krishnan/processed';
fMRIRun = 'rfMRI_VERBGEN_AP';

% Load surfaces
MidthickSurfs{1} = fullfile(BaseDir, Subject, 'MNINonLinear/fsaverage_LR32k', [Subject '.L.midthickness.32k_fs_LR.surf.gii']);
MidthickSurfs{2} = fullfile(BaseDir, Subject, 'MNINonLinear/fsaverage_LR32k', [Subject '.R.midthickness.32k_fs_LR.surf.gii']);

% Load dtseries (MSMSulc registration; Krishnan data is T1w-only so no MSMAll)
C = ft_read_cifti_mod(fullfile(BaseDir, Subject, 'MNINonLinear/Results', fMRIRun, [fMRIRun '_Atlas_hp2000_clean.dtseries.nii']));


%load([Subdir '/func/rest/ConcatenatedCiftis/FD.mat']);
C.data = single(C.data); % remove high motion volumes, convert to single type;

% Output directory
OutDir = fullfile(BaseDir, Subject, 'mshbm_output');
mkdir(OutDir);

% PFM with MS-HBM:

PriorWeight = 1; % this controls how much weight the spatial priors impose
Smoothness = 10; % controls how likely neighboring vertices are to belong to the same network
pfm_mshbm(C, MidthickSurfs, OutDir, PriorWeight, Smoothness, Params, Paths);
addpath(genpath(Paths{2})); % pfm_mshbm removes Paths{2} at exit; restore it

% Calculate the network sizes:

D = ft_read_cifti_mod(fullfile(OutDir, 'MS-HBM_FunctionalNetworks_VertexWiseThresh0.01_w1_c10.dlabel.nii'));
VA= ft_read_cifti_mod(fullfile(OutDir, 'MS-HBM_FunctionalNetworks_VertexWiseThresh0.01_w1_c10.dtseries.nii'));
Structures = {'CORTEX_LEFT','CORTEX_RIGHT'}; % in this case, cortex only.
NetworkSize = pfm_calculate_network_size(D,VA,Structures);

close all; % blank slate
H = figure; % prellocate parent figure
set(H,'position',[1 1 325 500]); hold;


% unique functional networks;
uCi = unique(nonzeros(D.data));


for i = 1:length(uCi)
    Tmp = nan(1,length(Params.NetworkLabels));
    Tmp(i) = NetworkSize(i);
    barh(Tmp,'FaceColor',Params.NetworkColors(i,:));
    text((NetworkSize(i)+0.1),i,[num2str(NetworkSize(i),3) '%']);
end

% make it pretty;
yticklabels(Params.NetworkLabels); 
yticks(1:length(uCi)); ylim([0 21]);
xlim([0 25]); xticks(0:5:25);
set(gca,'fontname','arial','fontsize',10,'TickLength',[0 0],'TickLabelInterpreter','none');
xlabel('% of Cortical Surface');