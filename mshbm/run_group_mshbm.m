function run_group_mshbm(varargin)
% run_group_mshbm  Run MS-HBM individual parcellation + network-size export for
% every analysed verb-gen subject (n = 144: DLD/BL, TD/BT, HSL/BH).
%
% Reads the subject list from the canonical analysis table
% dat_verbgen_analysis_144.csv and calls run_subject_mshbm for each. One
% subject failing does not abort the batch. With SkipIfDone (default true) the
% (slow) EM fit is skipped for subjects whose output already exists and only
% the network-size CSV is refreshed -- so the 37 subjects fitted previously are
% reused and only the 107 newly pre-processed subjects are actually fitted.
%
% Variant selects which run to parcellate ('full' = 25-vol-cut
% rfMRI_VERBGEN_AP_full, 'icafix' = all-vols rfMRI_VERBGEN_AP); run once per
% variant to build the two comparable network-size sets.
%
% Usage:
%   run_group_mshbm                                 % full variant, skip done
%   run_group_mshbm('Variant', 'icafix')            % icafix variant
%   run_group_mshbm('SkipIfDone', false)            % force re-fit even if done
%
% ## Author: Han Wang

p = inputParser;
addParameter(p, 'SkipIfDone', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'Variant', 'full', @(x) ischar(x) || isstring(x));
addParameter(p, 'Exclude', {}, @(x) iscell(x) || isstring(x) || ischar(x));
parse(p, varargin{:});
SkipIfDone = logical(p.Results.SkipIfDone);
Variant    = char(p.Results.Variant);
Exclude    = string(p.Results.Exclude);   % subject ids to skip, e.g. "sub-587BH"

ProjectDir = '/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen';
ListFile   = '/home/hanwang/Documents/Data/verb_gen_krishnan/behavioural_scq_sdq/dat_verbgen_analysis_144.csv';

% ------------------------------------------------------------
% Build subject list from the canonical analysis table (all 144)
% ------------------------------------------------------------
T = readtable(ListFile, 'VariableNamingRule', 'preserve');
subjects = "sub-" + string(T.code);
if ~isempty(Exclude)
    drop = ismember(subjects, Exclude);
    if any(drop)
        fprintf('Excluding %d subject(s): %s\n', sum(drop), strjoin(subjects(drop), ', '));
    end
    subjects = subjects(~drop);
end
fprintf('Found %d target subjects (variant=%s).\n', numel(subjects), Variant);

% ------------------------------------------------------------
% Loop
% ------------------------------------------------------------
nSub = numel(subjects);
ok = false(nSub, 1);
errmsg = strings(nSub, 1);

for i = 1:nSub
    sub = char(subjects(i));
    fprintf('\n========== [%d/%d] %s ==========\n', i, nSub, sub);
    try
        run_subject_mshbm(sub, 'SkipIfDone', SkipIfDone, 'Variant', Variant);
        ok(i) = true;
    catch ME
        errmsg(i) = string(ME.message);
        fprintf(2, 'FAILED %s: %s\n', sub, ME.message);
    end
end

% ------------------------------------------------------------
% Summary
% ------------------------------------------------------------
fprintf('\n================ SUMMARY ================\n');
fprintf('Succeeded: %d/%d\n', sum(ok), nSub);
if any(~ok)
    fprintf('Failed subjects:\n');
    failIdx = find(~ok);
    for k = 1:numel(failIdx)
        fprintf('  %s : %s\n', subjects(failIdx(k)), errmsg(failIdx(k)));
    end
end

% Write a small run log to the variant-specific results dir
switch Variant
    case 'full';   CsvSub = 'network_size_full';
    case 'icafix'; CsvSub = 'network_size_icafix';
    otherwise;     CsvSub = 'network_size';
end
ResultsDir = fullfile(ProjectDir, 'results', CsvSub);
if ~exist(ResultsDir, 'dir'); mkdir(ResultsDir); end
LogT = table(subjects, ok, errmsg, 'VariableNames', {'subject', 'succeeded', 'error'});
writetable(LogT, fullfile(ResultsDir, 'run_group_mshbm_log.csv'));
fprintf('Wrote run log: %s\n', fullfile(ResultsDir, 'run_group_mshbm_log.csv'));

end
