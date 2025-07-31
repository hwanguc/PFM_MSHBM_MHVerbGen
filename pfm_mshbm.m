function pfm_mshbm(C,MidthickSurfs,OutDir,PriorWeight,Smoothness,Params,Paths)
% cjl; cjl2007@med.cornell.edu;

rng(44); % for reproducibility.
warning ('off','all'); % annoying warnings;

% make output parent directories ;
system(['mkdir -p ' OutDir '/Test/priors/']);
system(['mkdir -p ' OutDir '/Training/priors/']);
save([OutDir '/Training/priors/Params_Final'],'Params');
save([OutDir '/Test/priors/Params_Final'],'Params');

% these are hard set folder names;
system(['mkdir -p ' OutDir '/Test/data_list/fMRI_list']);
system(['mkdir -p ' OutDir '/Test/data_list/censor_list']);

% adjust the paths;
rmpath(genpath(Paths{2}));
addpath(genpath(Paths{1}));

% make a temp. dir.;
mkdir([OutDir '/Tmp']);

% write out the temporary cifti file;
ft_write_cifti_mod([OutDir '/Tmp/Data'],C);

% identify good volumes;
GoodVols = double(ones(size(C.data,2),1));
save([OutDir '/Tmp/GoodVolumes.txt'],"GoodVols","-ascii")

% write out the .txt files;
system(['echo ' OutDir '/Tmp/GoodVolumes.txt >> ' OutDir '/Test/data_list/censor_list/sub1_sess1.txt']);
system(['echo ' OutDir '/Tmp/Data.dtseries.nii >> ' OutDir '/Test/data_list/fMRI_list/sub1_sess1.txt']);

% adjust the paths;
rmpath(genpath(Paths{1}));
addpath(genpath(Paths{2}));

% generate FC profiles using modified CBIG function;
CBIG_MSHBM_generate_profiles2('fs_LR_900','fs_LR_32k',[OutDir '/Test'],'1','1','0');
system(['rm -rf ' OutDir '/Tmp']); % remove the temporary directory;

% write out the paths to the FC profiles
system(['mkdir -p ' OutDir '/Test/profile_list/test_set/']);
system(['echo ' OutDir '/Test/profiles/sub1/sess1/sub1_sess1_fs_LR_32k_roifs_LR_900.surf2surf_profile_1.mat >> ' OutDir '/Test/profile_list/test_set/sess1.txt']);
system(['echo ' OutDir '/Test/profiles/sub1/sess1/sub1_sess1_fs_LR_32k_roifs_LR_900.surf2surf_profile_2.mat >> ' OutDir '/Test/profile_list/test_set/sess2.txt']);

% define output file name & run MS-HBM;
OutFile = ['MS-HBM_FunctionalNetworks_VertexWiseThresh0.01_w' num2str(PriorWeight) '_c' num2str(Smoothness)];
[lh,rh] = CBIG_MSHBM_generate_individual_parcellation([OutDir '/Test'],'fs_LR_32k','2','20','1',num2str(PriorWeight),num2str(Smoothness));

% adjust paths
rmpath(genpath(Paths{2}));
addpath(genpath(Paths{1}));

O = C; % preallocate the output;
O.data = zeros(size(C.data,1),1);

CorticalParcellation = [lh;rh]; % combine left and right

% identify subcortical structures;
BrainStructure = C.brainstructure;
CorticalParcellation(BrainStructure < 0) = [];
BrainStructure(BrainStructure < 0) = [];
SubcortexIdx = BrainStructure > 2;

% preallocate this variable;
Ci_ts = zeros(size(C.data,2),20);

% sweep the
% networks;
for i = 1:20
    Ci_ts(:,i) = mean(C.data(CorticalParcellation==i,:));
end
    
% calculate correlations with cortical systems 
r = corr(C.data(SubcortexIdx,:)',Ci_ts);
[r,SubcorticalParcellation] = max(r,[],2);
SubcorticalParcellation(r < 0.1) = 0;

% combine MS-HBM cortical parcellation
% and tack on subcortical winner-take-all assignments
O.data = [CorticalParcellation ; SubcorticalParcellation];

% write out the MSHBM network assignments ;
ft_write_cifti_mod([OutDir '/' OutFile],O);


% write out the first network;
system(['echo ' char(Params.NetworkLabels{1}) ' > ' OutDir '/LabelListFile.txt']);
system(['echo 1 ' num2str(round(Params.NetworkColors(1,1)*255)) ' ' num2str(round(Params.NetworkColors(1,2)*255)) ' ' num2str(round(Params.NetworkColors(1,3)*255)) ' 255 >> ' OutDir '/LabelListFile.txt ']);

% sweep through the networks;
for i = 2:length(Params.NetworkLabels)
    
    system(['echo ' char(Params.NetworkLabels{i}) ' >> ' OutDir '/LabelListFile.txt ']);
    system(['echo ' num2str(i) ' ' num2str(round(Params.NetworkColors(i,1)*255)) ' ' num2str(round(Params.NetworkColors(i,2)*255)) ' ' num2str(round(Params.NetworkColors(i,3)*255)) ' 255 >> ' OutDir '/LabelListFile.txt']);
    
end

% make dense label file + network borders;
system(['wb_command -cifti-label-import ' OutDir '/' OutFile '.dtseries.nii ' OutDir '/LabelListFile.txt ' OutDir '/' OutFile '.dlabel.nii -discard-others']);
system(['wb_command -cifti-label-to-border ' OutDir '/' OutFile '.dlabel.nii -border ' MidthickSurfs{1} ' ' OutDir '/' OutFile '.L.border']); % LH
system(['wb_command -cifti-label-to-border ' OutDir '/' OutFile '.dlabel.nii -border ' MidthickSurfs{2} ' ' OutDir '/' OutFile '.R.border']); % RH

% remove intermediate files;
system(['rm -rf ' OutDir '/T*']);
system(['rm ' OutDir '/LabelListFile.txt']);

end

