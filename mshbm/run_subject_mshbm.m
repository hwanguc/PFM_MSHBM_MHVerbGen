function sizeFull = run_subject_mshbm(Subject, varargin)
% run_subject_mshbm  Map individual functional networks with MS-HBM for one
% subject and save the cortical network sizes to a per-subject CSV under
% <ProjectDir>/results/network_size/.
%
% Usage:
%   run_subject_mshbm('sub-509BT')
%   run_subject_mshbm('sub-509BT', 'SkipIfDone', true)
%
%   SkipIfDone (default false): if the MS-HBM dlabel already exists, skip the
%   (slow) EM fit and just (re)compute the network sizes from the existing
%   output. Useful for regenerating CSVs without re-running the parcellation.
%
% Returns sizeFull: a 21x1 vector of % cortical surface per network
% (network id 1..21, 0 for networks absent in this subject).
%
% ## Author: Han Wang

if nargin < 1 || isempty(Subject)
    Subject = 'sub-509BT';   % default for interactive use
end
p = inputParser;
addParameter(p, 'SkipIfDone', false, @(x) islogical(x) || isnumeric(x));
parse(p, varargin{:});
SkipIfDone = logical(p.Results.SkipIfDone);

% ============================================================
% Paths
% ============================================================
ProjectDir = '/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen';
Paths{1} = fullfile(ProjectDir, 'res0urces/read_write_cifti'); % ft_read/write cifti
Paths{2} = fullfile(ProjectDir, 'res0urces/helper_functions'); % MS-HBM helper functions
Paths{3} = '~/Apps/Utils/CBIG-master';                         % Thomas Yeo's functions (modified)

addpath(genpath(Paths{1}));
addpath(genpath(Paths{2}));
addpath(genpath(Paths{3}));

load('MSHBM-Priors.mat', 'Params');  % spatial priors + network labels/colors

% ============================================================
% Subject data (Krishnan et al. 2021, adapted HCP pipeline; MSMSulc, no MSMAll)
% NB: preprocessed data was moved to /media/hanwang/Data to save space.
% ============================================================
BaseDir = '/media/hanwang/Data/Data/ucl/gos_ich/verb_gen_krishnan/processed';
fMRIRun = 'rfMRI_VERBGEN_AP';

MidthickSurfs{1} = fullfile(BaseDir, Subject, 'MNINonLinear/fsaverage_LR32k', [Subject '.L.midthickness.32k_fs_LR.surf.gii']);
MidthickSurfs{2} = fullfile(BaseDir, Subject, 'MNINonLinear/fsaverage_LR32k', [Subject '.R.midthickness.32k_fs_LR.surf.gii']);

OutDir = fullfile(BaseDir, Subject, 'mshbm_output');
DlabelFile  = fullfile(OutDir, 'MS-HBM_FunctionalNetworks_VertexWiseThresh0.01_w1_c10.dlabel.nii');
DtseriesNet = fullfile(OutDir, 'MS-HBM_FunctionalNetworks_VertexWiseThresh0.01_w1_c10.dtseries.nii');

% ============================================================
% PFM with MS-HBM (skip the EM fit if requested and output already exists)
% ============================================================
if SkipIfDone && exist(DlabelFile, 'file')
    fprintf('[%s] MS-HBM output exists; skipping EM fit.\n', Subject);
else
    C = ft_read_cifti_mod(fullfile(BaseDir, Subject, 'MNINonLinear/Results', fMRIRun, [fMRIRun '_Atlas_hp2000_clean.dtseries.nii']));
    C.data = single(C.data);  % convert to single

    mkdir(OutDir);

    PriorWeight = 1;  % weight of the spatial priors
    Smoothness  = 10; % spatial smoothness (neighbouring vertices share a network)
    pfm_mshbm(C, MidthickSurfs, OutDir, PriorWeight, Smoothness, Params, Paths);
    addpath(genpath(Paths{2})); % pfm_mshbm removes Paths{2} at exit; restore it
end

% ============================================================
% Calculate cortical network sizes
% ============================================================
D  = ft_read_cifti_mod(DlabelFile);
VA = ft_read_cifti_mod(DtseriesNet);
Structures  = {'CORTEX_LEFT', 'CORTEX_RIGHT'};  % cortex only
NetworkSize = pfm_calculate_network_size(D, VA, Structures);  % size per present network (ascending id)

% Map sizes onto the full 21-network vector (0 for networks absent in subject).
% pfm_calculate_network_size returns one value per unique non-zero network id,
% in ascending id order, so index by uCi.
uCi  = unique(nonzeros(D.data));
nNet = numel(Params.NetworkLabels);
sizeFull = zeros(nNet, 1);
sizeFull(uCi) = NetworkSize(:);

% ============================================================
% Save per-subject CSV: subject, network_id, network_label, network_size_pct
% ============================================================
ResultsDir = fullfile(ProjectDir, 'results', 'network_size');
if ~exist(ResultsDir, 'dir'); mkdir(ResultsDir); end

T = table( ...
    repmat(string(Subject), nNet, 1), ...
    (1:nNet)', ...
    string(Params.NetworkLabels(:)), ...
    sizeFull, ...
    'VariableNames', {'subject', 'network_id', 'network_label', 'network_size_pct'});

csvPath = fullfile(ResultsDir, [Subject '_networksize.csv']);
writetable(T, csvPath);
fprintf('[%s] Saved network sizes: %s\n', Subject, csvPath);

end
