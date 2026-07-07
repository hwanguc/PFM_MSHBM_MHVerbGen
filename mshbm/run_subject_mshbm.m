function sizeFull = run_subject_mshbm(Subject, varargin)
% run_subject_mshbm  Map individual functional networks with MS-HBM for one
% subject and save the cortical network sizes to a per-subject CSV under
% <ProjectDir>/results/network_size/.
%
% Usage:
%   run_subject_mshbm('sub-509BT')
%   run_subject_mshbm('sub-509BT', 'SkipIfDone', true)
%   run_subject_mshbm('sub-509BT', 'Variant', 'icafix')
%
%   SkipIfDone (default false): if the MS-HBM dlabel already exists, skip the
%   (slow) EM fit and just (re)compute the network sizes from the existing
%   output. Useful for regenerating CSVs without re-running the parcellation.
%
%   Variant (default 'full'): which re-extraction of the run to parcellate.
%     'full'   -> run 'rfMRI_VERBGEN_AP_full' (session minus first 25 vols),
%                 output dir <sub>/mshbm_output, CSVs results/network_size_full/.
%     'icafix' -> run 'rfMRI_VERBGEN_AP' (standard ICA+FIX hp2000_clean, all
%                 vols), output dir <sub>/mshbm_output_pre25vol, CSVs
%                 results/network_size_icafix/.
%   Running both gives two comparable network-size sets (25-vol-cut vs not).
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
addParameter(p, 'Variant', 'full', @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
SkipIfDone = logical(p.Results.SkipIfDone);
Variant    = char(p.Results.Variant);

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
% NB: preprocessed data + MS-HBM outputs live on the 2T internal drive (2026-07),
% whose mount path contains a space ("Data 001 2T"). pfm_mshbm/CBIG shell out to
% `mkdir`/etc. with UNQUOTED paths, which word-split on the space and fail with
% "Permission denied". So point BaseDir at a no-space symlink that resolves to the
% 2T drive:  ~/verbgen_processed -> "/run/media/hanwang/Data 001 2T/.../processed".
% (Create it once with:
%   ln -sfn "/run/media/hanwang/Data 001 2T/hanwang/Documents/Data/verb_gen_krishnan/processed" ~/verbgen_processed )
% ============================================================
BaseDir = '/home/hanwang/verbgen_processed';

% Two comparable re-extractions of the run (see Variant in the help above):
%   'full'   = 'rfMRI_VERBGEN_AP_full' (session minus first 25 vols; the noise-
%              cancelling headphones were still adapting to the scanner
%              background noise during that window) -> dir 'mshbm_output'.
%   'icafix' = 'rfMRI_VERBGEN_AP' (standard ICA+FIX hp2000_clean, all vols)
%              -> dir 'mshbm_output_pre25vol'.
% The rest-only '_rest' run is used by the connectivity pipeline, not here.
switch Variant
    case 'full'
        fMRIRun = 'rfMRI_VERBGEN_AP_full';
        OutName = 'mshbm_output';
        CsvSub  = 'network_size_full';
    case 'icafix'
        fMRIRun = 'rfMRI_VERBGEN_AP';
        OutName = 'mshbm_output_pre25vol';
        CsvSub  = 'network_size_icafix';
    otherwise
        error('run_subject_mshbm:Variant', ...
              'Unknown Variant "%s" (use ''full'' or ''icafix'').', Variant);
end

MidthickSurfs{1} = fullfile(BaseDir, Subject, 'MNINonLinear/fsaverage_LR32k', [Subject '.L.midthickness.32k_fs_LR.surf.gii']);
MidthickSurfs{2} = fullfile(BaseDir, Subject, 'MNINonLinear/fsaverage_LR32k', [Subject '.R.midthickness.32k_fs_LR.surf.gii']);

OutDir = fullfile(BaseDir, Subject, OutName);
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
ResultsDir = fullfile(ProjectDir, 'results', CsvSub);
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
