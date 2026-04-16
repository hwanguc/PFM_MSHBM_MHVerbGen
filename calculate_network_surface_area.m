function calculate_network_surface_area(dlabel_path, va_path, network_labels, output_csv)

% Load the dense label file (network parcellation)
D = ft_read_cifti_mod(dlabel_path);

% Load the vertex area file
VA = ft_read_cifti_mod(va_path);

% Extract labels
all_labels = unique(D.data);
all_labels(all_labels == 0) = []; % remove unlabeled

% Initialize output
absolute_area = zeros(length(all_labels), 1);
percent_area  = zeros(length(all_labels), 1);

% Total cortical surface area (exclude unlabeled)
total_area = sum(VA.data(D.data > 0));

for i = 1:length(all_labels)
    idx = D.data == all_labels(i);
    absolute_area(i) = sum(VA.data(idx));
    percent_area(i)  = 100 * absolute_area(i) / total_area;
end

% Prepare table
if nargin < 3 || isempty(network_labels)
    network_labels = cellstr("Network_" + string(all_labels));
end

T = table(network_labels(:), all_labels(:), absolute_area, percent_area, ...
    'VariableNames', {'Label', 'NetworkID', 'SurfaceArea_mm2', 'SurfaceAreaPercent'});

% Write to CSV
writetable(T, output_csv);

fprintf('Network surface areas saved to %s\n', output_csv);
end