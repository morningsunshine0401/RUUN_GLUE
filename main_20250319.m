%% Script to compare raw and refined/KF data from the same JSON file
% No longer using ROS bag (db3) file for ground truth

clear all; close all; clc;

%% (1) Preparations

% Edit this path to match your data:
jsonPath = 'pose_estimation_MEKF.json';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create a "Matlab_results" folder and a subfolder with today's date to save plots
baseFolder = 'Matlab_results';
if ~exist(baseFolder, 'dir')
    mkdir(baseFolder);
end

dateStr = datestr(now,'yyyy-mm-dd');
testCaseFolder = fullfile(baseFolder, dateStr);

% Extract part of the file name for your test case
[~, jsonBase, ~] = fileparts(jsonPath);
testCaseFolder = fullfile(testCaseFolder, jsonBase);

if ~exist(testCaseFolder, 'dir')
    mkdir(testCaseFolder);
end

saveFolder = testCaseFolder;

%% (2) Load the JSON pose estimation data
jsonText = fileread(jsonPath);
algData = jsondecode(jsonText);
N_json = numel(algData);

fprintf("Loaded %d frames from JSON file\n", N_json);

%% (3) Frame selection for plotting
% Specify the desired number of frames to plot
desiredFrameCount = 600;
% Automatically select frames from 1 up to the lesser of desiredFrameCount or the available JSON frames
selectedFrames = 1:min(desiredFrameCount, N_json);
if max(selectedFrames) > N_json
    error('Selected frames exceed the available JSON frames.');
end

%% (4) Extract both raw and KF data from JSON
% We'll store the rotation and translation from both data types
R_tgt_in_cam_raw = cell(N_json, 1);
t_tgt_in_cam_raw = nan(N_json, 3);
R_tgt_in_cam_kf = cell(N_json, 1);
t_tgt_in_cam_kf = nan(N_json, 3);
jsonTimes = nan(N_json, 1);

% Define field names for both data types
rawRotField = 'raw_pnp_rotation';
rawTransField = 'raw_pnp_translation';
kfRotField = 'kf_rotation_matrix';
kfTransField = 'kf_translation_vector';

validEntries = 0;
for i = 1:N_json
    thisEntry = algData{i};
    
    % Store timestamp if available
    if isfield(thisEntry, 'timestamp')
        jsonTimes(i) = thisEntry.timestamp;
    end
    
    % Check if both raw and KF fields exist in this entry
    hasRawData = isfield(thisEntry, rawRotField) && isfield(thisEntry, rawTransField);
    hasKfData = isfield(thisEntry, kfRotField) && isfield(thisEntry, kfTransField);
    
    if hasRawData && hasKfData
        validEntries = validEntries + 1;
        
        % Process raw data
        R_raw = thisEntry.(rawRotField);
        t_raw = thisEntry.(rawTransField);
        
        % Process KF data
        R_kf = thisEntry.(kfRotField);
        t_kf = thisEntry.(kfTransField);
        
        % Reshape and store raw rotation matrix
        if ~isempty(R_raw) && numel(R_raw) == 9
            R_tgt_in_cam_raw{i} = reshape(R_raw, [3,3]);
        else
            R_tgt_in_cam_raw{i} = NaN(3,3);
        end
        
        % Store raw translation vector
        if ~isempty(t_raw) && numel(t_raw) == 3
            t_tgt_in_cam_raw(i,:) = t_raw(:)';
        end
        
        % Reshape and store KF rotation matrix
        if ~isempty(R_kf) && numel(R_kf) == 9
            R_tgt_in_cam_kf{i} = reshape(R_kf, [3,3]);
        else
            R_tgt_in_cam_kf{i} = NaN(3,3);
        end
        
        % Store KF translation vector
        if ~isempty(t_kf) && numel(t_kf) == 3
            t_tgt_in_cam_kf(i,:) = t_kf(:)';
        end
    end
end

fprintf("Found %d valid entries with both raw and KF data\n", validEntries);

%% (5) Compute Euler angles for both data types
eul_raw = zeros(N_json, 3);
eul_kf = zeros(N_json, 3);

for i = 1:N_json
    % Check that the cell is not empty before accessing its elements
    if ~isempty(R_tgt_in_cam_raw{i}) && ~isempty(R_tgt_in_cam_kf{i}) && ...
       ~isnan(R_tgt_in_cam_raw{i}(1,1)) && ~isnan(R_tgt_in_cam_kf{i}(1,1))
        eul_raw(i,:) = rotm2eul(R_tgt_in_cam_raw{i}, 'XYZ');
        eul_kf(i,:) = rotm2eul(R_tgt_in_cam_kf{i}, 'XYZ');
    else
        eul_raw(i,:) = [NaN, NaN, NaN];
        eul_kf(i,:) = [NaN, NaN, NaN];
    end
end

%% (6) Plot X/Y/Z position comparison for selected frames
figure('Name', 'Position Comparison (Raw vs KF)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
hold on; grid on;
plot(selectedFrames, t_tgt_in_cam_raw(selectedFrames,1), 'b--', 'LineWidth', 1.7);
plot(selectedFrames, t_tgt_in_cam_kf(selectedFrames,1), 'b-', 'LineWidth', 1.7);
plot(selectedFrames, t_tgt_in_cam_raw(selectedFrames,2), 'g--', 'LineWidth', 1.7);
plot(selectedFrames, t_tgt_in_cam_kf(selectedFrames,2), 'g-', 'LineWidth', 1.7);
plot(selectedFrames, t_tgt_in_cam_raw(selectedFrames,3), 'r--', 'LineWidth', 1.7);
plot(selectedFrames, t_tgt_in_cam_kf(selectedFrames,3), 'r-', 'LineWidth', 1.7);
set(gca, 'FontSize', 14);
xlabel('Frame', 'FontSize', 16); 
ylabel('Position (m)', 'FontSize', 16);
legend({'Raw X', 'KF X', 'Raw Y', 'KF Y', 'Raw Z', 'KF Z'}, 'Location', 'best');
title('Position Comparison: Raw vs KF', 'FontSize', 18);
hold off;

print(gcf, fullfile(saveFolder, ...
    [jsonBase '_PositionComparison.png']), '-dpng', '-r300');

%% (7) Orientation comparison for selected frames (Euler angles)
figure('Name', 'Orientation Comparison (Raw vs KF)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
hold on; grid on;
plot(selectedFrames, rad2deg(eul_raw(selectedFrames,1)), 'b--', 'LineWidth', 1.7);
plot(selectedFrames, rad2deg(eul_kf(selectedFrames,1)), 'b-', 'LineWidth', 1.7);
plot(selectedFrames, rad2deg(eul_raw(selectedFrames,2)), 'g--', 'LineWidth', 1.7);
plot(selectedFrames, rad2deg(eul_kf(selectedFrames,2)), 'g-', 'LineWidth', 1.7);
plot(selectedFrames, rad2deg(eul_raw(selectedFrames,3)), 'm--', 'LineWidth', 1.7);
plot(selectedFrames, rad2deg(eul_kf(selectedFrames,3)), 'm-', 'LineWidth', 1.7);
set(gca, 'FontSize', 14);
xlabel('Frame', 'FontSize', 16); 
ylabel('Angle (deg)', 'FontSize', 16);
legend({'Raw Roll', 'KF Roll', 'Raw Pitch', 'KF Pitch', 'Raw Yaw', 'KF Yaw'}, 'Location', 'best');
title('Orientation Comparison: Raw vs KF', 'FontSize', 18);
hold off;

print(gcf, fullfile(saveFolder, ...
    [jsonBase '_OrientationComparison.png']), '-dpng', '-r300');

%% (8) Compute differences between raw and KF data
pos_diff = t_tgt_in_cam_raw - t_tgt_in_cam_kf;
eul_diff = eul_raw - eul_kf;
deg_diff = rad2deg(eul_diff);

% Restrict to selected frames for plotting
pos_diff_plot = pos_diff(selectedFrames,:);
deg_diff_plot = deg_diff(selectedFrames,:);

%% (9) Plot position differences
figure('Name', 'Position Difference (Raw - KF)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
subplot(3,1,1); hold on; grid on;
plot(selectedFrames, pos_diff_plot(:,1), 'r-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Diff X (m)');
title('Position Difference in X (Raw - KF)');
subplot(3,1,2); hold on; grid on;
plot(selectedFrames, pos_diff_plot(:,2), 'g-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Diff Y (m)');
title('Position Difference in Y (Raw - KF)');
subplot(3,1,3); hold on; grid on;
plot(selectedFrames, pos_diff_plot(:,3), 'b-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Diff Z (m)');
title('Position Difference in Z (Raw - KF)');

print(gcf, fullfile(saveFolder, ...
    [jsonBase '_PositionDifference.png']), '-dpng', '-r300');

%% (10) Plot orientation differences
figure('Name', 'Orientation Difference (Raw - KF)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
subplot(3,1,1); hold on; grid on;
plot(selectedFrames, deg_diff_plot(:,1), 'r-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Roll Diff (deg)');
title('Orientation Difference in Roll (Raw - KF)');
subplot(3,1,2); hold on; grid on;
plot(selectedFrames, deg_diff_plot(:,2), 'g-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Pitch Diff (deg)');
title('Orientation Difference in Pitch (Raw - KF)');
subplot(3,1,3); hold on; grid on;
plot(selectedFrames, deg_diff_plot(:,3), 'b-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Yaw Diff (deg)');
title('Orientation Difference in Yaw (Raw - KF)');

print(gcf, fullfile(saveFolder, ...
    [jsonBase '_OrientationDifference.png']), '-dpng', '-r300');

%% (11) Calculate and display statistics
% Calculate statistics for position differences (in mm for better readability)
pos_diff_mm = pos_diff * 1000; % Convert to mm
pos_stats = struct();
pos_stats.mean = mean(pos_diff_mm, 'omitnan');
pos_stats.std = std(pos_diff_mm, 'omitnan');
pos_stats.max = max(abs(pos_diff_mm), [], 'omitnan');
pos_stats.rms = sqrt(mean(pos_diff_mm.^2, 'omitnan'));

% Calculate statistics for orientation differences (in degrees)
ori_stats = struct();
ori_stats.mean = mean(deg_diff, 'omitnan');
ori_stats.std = std(deg_diff, 'omitnan');
ori_stats.max = max(abs(deg_diff), [], 'omitnan');
ori_stats.rms = sqrt(mean(deg_diff.^2, 'omitnan'));

% Display statistics
fprintf('\n--- Position Difference Statistics (Raw - KF) in mm ---\n');
fprintf('Mean: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.mean(1), pos_stats.mean(2), pos_stats.mean(3));
fprintf('Std Dev: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.std(1), pos_stats.std(2), pos_stats.std(3));
fprintf('Max Abs: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.max(1), pos_stats.max(2), pos_stats.max(3));
fprintf('RMS: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.rms(1), pos_stats.rms(2), pos_stats.rms(3));

fprintf('\n--- Orientation Difference Statistics (Raw - KF) in degrees ---\n');
fprintf('Mean: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.mean(1), ori_stats.mean(2), ori_stats.mean(3));
fprintf('Std Dev: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.std(1), ori_stats.std(2), ori_stats.std(3));
fprintf('Max Abs: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.max(1), ori_stats.max(2), ori_stats.max(3));
fprintf('RMS: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.rms(1), ori_stats.rms(2), ori_stats.rms(3));

%% (12) Optional: Calculate and plot distance measurements
% Calculate distances from camera to target for both raw and KF data
raw_distances = vecnorm(t_tgt_in_cam_raw, 2, 2);
kf_distances = vecnorm(t_tgt_in_cam_kf, 2, 2);
distance_diff = raw_distances - kf_distances;

% Plot distance comparison
figure('Name', 'Distance Comparison (Raw vs KF)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 400]);
hold on; grid on;
plot(selectedFrames, raw_distances(selectedFrames), 'b--', 'LineWidth', 1.5);
plot(selectedFrames, kf_distances(selectedFrames), 'r-', 'LineWidth', 1.5);
plot(selectedFrames, distance_diff(selectedFrames), 'k-.', 'LineWidth', 1.5);
xlabel('Frame', 'FontSize', 14);
ylabel('Distance (m)', 'FontSize', 14);
title('Distance Comparison (Raw vs KF)', 'FontSize', 16);
legend({'Raw Distance', 'KF Distance', 'Difference (Raw - KF)'}, 'Location', 'best');
hold off;

print(gcf, fullfile(saveFolder, ...
    [jsonBase '_DistanceComparison.png']), '-dpng', '-r300');

% Calculate distance statistics
dist_diff_mm = distance_diff * 1000; % Convert to mm
dist_stats = struct();
dist_stats.mean = mean(dist_diff_mm, 'omitnan');
dist_stats.std = std(dist_diff_mm, 'omitnan');
dist_stats.max = max(abs(dist_diff_mm), [], 'omitnan');
dist_stats.rms = sqrt(mean(dist_diff_mm.^2, 'omitnan'));

fprintf('\n--- Distance Difference Statistics (Raw - KF) in mm ---\n');
fprintf('Mean: %.2f\n', dist_stats.mean);
fprintf('Std Dev: %.2f\n', dist_stats.std);
fprintf('Max Abs: %.2f\n', dist_stats.max);
fprintf('RMS: %.2f\n', dist_stats.rms);

disp("Finished plotting. All figures saved automatically in: " + saveFolder);
