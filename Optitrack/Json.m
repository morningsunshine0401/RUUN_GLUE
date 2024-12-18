clc; clear; close all;

% Load the .mat file
load('matlab_box.mat'); % Ensure this file is in the working directory

% Initialize storage for camera and target positions
numRows = numel(msgs_mat_box); % Number of structures in msgs_mat
data = struct();

% Loop through each structure
for i = 1:numRows
    % Access the 14x1 data from the structure (assuming a field exists)
    if isfield(msgs_mat_box(i), 'data') % Replace 'data' with the actual field name
        currentData = msgs_mat_box(i).data; 
        
        % Ensure it has at least 10 elements
        if numel(currentData) >= 10
            % Extract camera position (first 3 elements: X, Y, Z)
            camera_pos = currentData(1:3);
            
            % Extract target position (8th to 10th elements: X, Y, Z)
            target_pos = currentData(8:10);
            
            % Store in a structure
            data(i).camera_position = struct('X', camera_pos(1), 'Y', camera_pos(2), 'Z', camera_pos(3));
            data(i).target_position = struct('X', target_pos(1), 'Y', target_pos(2), 'Z', target_pos(3));
        else
            warning('Structure %d does not have enough elements. Skipping...', i);
        end
    else
        warning('Structure %d does not contain the required field. Skipping...', i);
    end
end

% Convert to JSON format
jsonData = jsonencode(data);

% Save to a file
fileID = fopen('extracted_positions.json', 'w');
fprintf(fileID, '%s', jsonData);
fclose(fileID);

disp('Data extraction complete. JSON saved as extracted_positions.json');
