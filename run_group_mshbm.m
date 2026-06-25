function run_group_mshbm(varargin)
% run_group_mshbm  Run MS-HBM individual parcellation + network-size export for
% every BL (DLD) and BT (control) subject in the verb-gen subsample.
%
% Reads the subject list from the behavioural spreadsheet, keeps codes ending
% in 'BL' (DLD) or 'BT' (TD/control), drops 513BT (failed QC, replaced by
% 675BT), and calls run_subject_mshbm for each. One subject failing does not
% abort the batch. Re-running skips the EM fit for subjects already done
% (SkipIfDone) and just refreshes their CSV.
%
% Usage:
%   run_group_mshbm                 % run/refresh all BL+BT subjects
%   run_group_mshbm('SkipIfDone', false)  % force re-fit even if output exists
%
% ## Author: Han Wang

p = inputParser;
addParameter(p, 'SkipIfDone', true, @(x) islogical(x) || isnumeric(x));
parse(p, varargin{:});
SkipIfDone = logical(p.Results.SkipIfDone);

ProjectDir = '/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen';
XlsxFile   = '/home/hanwang/Documents/Data/ucl/gos_ich/verb_gen_krishnan/behavioural_scq_sdq/dat_verbgen_scqsdq_subsample.xlsx';

% ------------------------------------------------------------
% Build subject list: BL (DLD) + BT (control), excluding 513BT
% ------------------------------------------------------------
T = readtable(XlsxFile, 'VariableNamingRule', 'preserve');
codes = string(T.code);
keep  = (endsWith(codes, 'BL') | endsWith(codes, 'BT')) & codes ~= "513BT";
codes = codes(keep);
subjects = "sub-" + codes;
fprintf('Found %d target subjects (BL+BT, excluding 513BT).\n', numel(subjects));

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
        run_subject_mshbm(sub, 'SkipIfDone', SkipIfDone);
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

% Write a small run log to results/network_size/
ResultsDir = fullfile(ProjectDir, 'results', 'network_size');
if ~exist(ResultsDir, 'dir'); mkdir(ResultsDir); end
LogT = table(subjects, ok, errmsg, 'VariableNames', {'subject', 'succeeded', 'error'});
writetable(LogT, fullfile(ResultsDir, 'run_group_mshbm_log.csv'));
fprintf('Wrote run log: %s\n', fullfile(ResultsDir, 'run_group_mshbm_log.csv'));

end
