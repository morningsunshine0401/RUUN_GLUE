%% Script to compare Kalman filtered data from two different pose estimation algorithms
% Compares KF results from two JSON files representing different algorithms on the same dataset

clear all; close all; clc;
%% (1) Preparations

% Edit these paths to match your two algorithm results:
jsonPath_A = 'VAPE_indoor.json';  % First algorithm
jsonPath_B = 'RTM_indoor.json';  % Second algorithm

% Alternative examples:
% jsonPath_A = 'VAPE_outdoor.json';
% jsonPath_B = 'pose_estimation_MEKF_20250320_6.json';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create a "Matlab_results" folder and a subfolder with today's date to save plots
baseFolder = 'Matlab_results';
if ~exist(baseFolder, 'dir')
    mkdir(baseFolder);
end

dateStr = datestr(now,'yyyy-mm-dd');
testCaseFolder = fullfile(baseFolder, dateStr);

% Extract parts of the file names for your test case
[~, jsonBase_A, ~] = fileparts(jsonPath_A);
[~, jsonBase_B, ~] = fileparts(jsonPath_B);
comparisonName = [jsonBase_A '_vs_' jsonBase_B];
testCaseFolder = fullfile(testCaseFolder, comparisonName);

if ~exist(testCaseFolder, 'dir')
    mkdir(testCaseFolder);
end

saveFolder = testCaseFolder;

%% (2) Load the JSON pose estimation data from both algorithms
% Load Algorithm A
jsonText_A = fileread(jsonPath_A);
algData_A = jsondecode(jsonText_A);
N_json_A = numel(algData_A);

% Load Algorithm B
jsonText_B = fileread(jsonPath_B);
algData_B = jsondecode(jsonText_B);
N_json_B = numel(algData_B);

fprintf("Loaded %d frames from Algorithm A JSON file\n", N_json_A);
fprintf("Loaded %d frames from Algorithm B JSON file\n", N_json_B);

%% (3) Frame selection for plotting
% Use the minimum number of frames available from both algorithms
N_frames = min(N_json_A, N_json_B);
desiredFrameCount = 560;
% Automatically select frames from 1 up to the lesser of desiredFrameCount or available frames
selectedFrames = 1:min(desiredFrameCount, N_frames);

fprintf("Using %d frames for comparison\n", length(selectedFrames));

%% (4) Extract KF data from both JSON files
% Algorithm A KF data
R_tgt_in_cam_A = cell(N_frames, 1);
t_tgt_in_cam_A = nan(N_frames, 3);
jsonTimes_A = nan(N_frames, 1);

% Algorithm B KF data
R_tgt_in_cam_B = cell(N_frames, 1);
t_tgt_in_cam_B = nan(N_frames, 3);
jsonTimes_B = nan(N_frames, 1);

% Define field names for KF data (modify these based on your JSON structure)
kfRotField = 'rotation_matrix';
kfTransField = 'position';

validKfEntries_A = 0;
validKfEntries_B = 0;

% Process Algorithm A data
for i = 1:N_frames
    thisEntry = algData_A{i};
    
    % Store timestamp if available
    if isfield(thisEntry, 'timestamp')
        jsonTimes_A(i) = thisEntry.timestamp;
    end

    % Process KF data
    hasKfData = isfield(thisEntry, kfRotField) && isfield(thisEntry, kfTransField);
    if hasKfData
        validKfEntries_A = validKfEntries_A + 1;

        R_kf = thisEntry.(kfRotField);
        t_kf = thisEntry.(kfTransField);

        % Reshape and store KF rotation matrix
        if ~isempty(R_kf) && numel(R_kf) == 9
            R_tgt_in_cam_A{i} = reshape(R_kf, [3,3]);
        else
            R_tgt_in_cam_A{i} = NaN(3,3);
        end

        % Store KF translation vector
        if ~isempty(t_kf) && numel(t_kf) == 3
            t_tgt_in_cam_A(i,:) = t_kf(:)';
        end
    else
        R_tgt_in_cam_A{i} = NaN(3,3);
    end
end

% Process Algorithm B data
for i = 1:N_frames
    %thisEntry = algData_B{i};
    thisEntry = algData_B(i);
    
    % Store timestamp if available
    if isfield(thisEntry, 'timestamp')
        jsonTimes_B(i) = thisEntry.timestamp;
    end

    % Process KF data
    hasKfData = isfield(thisEntry, kfRotField) && isfield(thisEntry, kfTransField);
    if hasKfData
        validKfEntries_B = validKfEntries_B + 1;

        R_kf = thisEntry.(kfRotField);
        t_kf = thisEntry.(kfTransField);

        % Reshape and store KF rotation matrix
        if ~isempty(R_kf) && numel(R_kf) == 9
            R_tgt_in_cam_B{i} = reshape(R_kf, [3,3]);
        else
            R_tgt_in_cam_B{i} = NaN(3,3);
        end

        % Store KF translation vector
        if ~isempty(t_kf) && numel(t_kf) == 3
            t_tgt_in_cam_B(i,:) = t_kf(:)';
        end
    else
        R_tgt_in_cam_B{i} = NaN(3,3);
    end
end

fprintf("Algorithm A: Found %d entries with KF data\n", validKfEntries_A);
fprintf("Algorithm B: Found %d entries with KF data\n", validKfEntries_B);

%% (5) Compute Euler angles for both algorithms
eul_A = zeros(N_frames, 3);
eul_B = zeros(N_frames, 3);

for i = 1:N_frames
    % Process Algorithm A rotation data
    if ~isempty(R_tgt_in_cam_A{i}) && ~all(isnan(R_tgt_in_cam_A{i}(:)))
        eul_A(i,:) = rotm2eul(R_tgt_in_cam_A{i}, 'XYZ');
    else
        eul_A(i,:) = [NaN, NaN, NaN];
    end

    % Process Algorithm B rotation data
    if ~isempty(R_tgt_in_cam_B{i}) && ~all(isnan(R_tgt_in_cam_B{i}(:)))
        eul_B(i,:) = rotm2eul(R_tgt_in_cam_B{i}, 'XYZ');
    else
        eul_B(i,:) = [NaN, NaN, NaN];
    end
end

%% (5.1) Compute quaternions for both algorithms
quat_A = nan(N_frames, 4);
quat_B = nan(N_frames, 4);

for i = 1:N_frames
    % Algorithm A rotation to quaternion
    if ~isempty(R_tgt_in_cam_A{i}) && ~all(isnan(R_tgt_in_cam_A{i}(:)))
        quat_A(i,:) = rotm2quat(R_tgt_in_cam_A{i});  % returns [w x y z]
    end

    % Algorithm B rotation to quaternion
    if ~isempty(R_tgt_in_cam_B{i}) && ~all(isnan(R_tgt_in_cam_B{i}(:)))
        quat_B(i,:) = rotm2quat(R_tgt_in_cam_B{i});  % returns [w x y z]
    end
end

%% (6) Plot X/Y/Z position comparison for selected frames
figure('Name', 'Position Comparison (Algorithm A vs B)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
hold on; grid on;

% Plot Algorithm A position data
plot(selectedFrames, t_tgt_in_cam_A(selectedFrames,1), 'b-', 'LineWidth', 1.7);
plot(selectedFrames, t_tgt_in_cam_A(selectedFrames,2), 'g-', 'LineWidth', 1.7);
plot(selectedFrames, t_tgt_in_cam_A(selectedFrames,3), 'r-', 'LineWidth', 1.7);

% Plot Algorithm B position data
plot(selectedFrames, t_tgt_in_cam_B(selectedFrames,1), 'b--', 'LineWidth', 1.7);
plot(selectedFrames, t_tgt_in_cam_B(selectedFrames,2), 'g--', 'LineWidth', 1.7);
plot(selectedFrames, t_tgt_in_cam_B(selectedFrames,3), 'r--', 'LineWidth', 1.7);

set(gca, 'FontSize', 14);
xlabel('Frame', 'FontSize', 16); 
ylabel('Position (m)', 'FontSize', 16);
legend({'Alg A X', 'Alg A Y', 'Alg A Z', 'Alg B X', 'Alg B Y', 'Alg B Z'}, 'Location', 'best');
title('Position Comparison: Algorithm A vs Algorithm B', 'FontSize', 18);
hold off;

print(gcf, fullfile(saveFolder, [comparisonName '_PositionComparison.png']), '-dpng', '-r300');

%% (7) Orientation comparison for selected frames (Euler angles)
figure('Name', 'Orientation Comparison (Algorithm A vs B)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
hold on; grid on;

% Plot Algorithm A orientation data
plot(selectedFrames, rad2deg(eul_A(selectedFrames,1)), 'b-', 'LineWidth', 1.7);
plot(selectedFrames, rad2deg(eul_A(selectedFrames,2)), 'g-', 'LineWidth', 1.7);
plot(selectedFrames, rad2deg(eul_A(selectedFrames,3)), 'm-', 'LineWidth', 1.7);

% Plot Algorithm B orientation data
plot(selectedFrames, rad2deg(eul_B(selectedFrames,1)), 'b--', 'LineWidth', 1.7);
plot(selectedFrames, rad2deg(eul_B(selectedFrames,2)), 'g--', 'LineWidth', 1.7);
plot(selectedFrames, rad2deg(eul_B(selectedFrames,3)), 'm--', 'LineWidth', 1.7);

set(gca, 'FontSize', 14);
xlabel('Frame', 'FontSize', 16); 
ylabel('Angle (deg)', 'FontSize', 16);
legend({'Alg A Roll', 'Alg A Pitch', 'Alg A Yaw', 'Alg B Roll', 'Alg B Pitch', 'Alg B Yaw'}, 'Location', 'best');
title('Orientation Comparison: Algorithm A vs Algorithm B', 'FontSize', 18);
hold off;

print(gcf, fullfile(saveFolder, [comparisonName '_OrientationComparison.png']), '-dpng', '-r300');

%% (7.1) Plot Quaternion Comparison (Algorithm A vs B)
figure('Name','Quaternion Comparison (Algorithm A vs B)', ...
       'NumberTitle','off', 'Position',[100, 100, 900, 800]);

% w-component
subplot(4,1,1); hold on; grid on;
plot(selectedFrames, quat_A(selectedFrames,1), 'b-', 'LineWidth', 1.7);
plot(selectedFrames, quat_B(selectedFrames,1), 'b--', 'LineWidth', 1.7);
ylabel('q_w', 'FontSize', 14);
legend('Alg A q_w','Alg B q_w','Location','best');

% x-component
subplot(4,1,2); hold on; grid on;
plot(selectedFrames, quat_A(selectedFrames,2), 'g-', 'LineWidth', 1.7);
plot(selectedFrames, quat_B(selectedFrames,2), 'g--', 'LineWidth', 1.7);
ylabel('q_x', 'FontSize', 14);
legend('Alg A q_x','Alg B q_x','Location','best');

% y-component
subplot(4,1,3); hold on; grid on;
plot(selectedFrames, quat_A(selectedFrames,3), 'r-', 'LineWidth', 1.7);
plot(selectedFrames, quat_B(selectedFrames,3), 'r--', 'LineWidth', 1.7);
ylabel('q_y', 'FontSize', 14);
legend('Alg A q_y','Alg B q_y','Location','best');

% z-component
subplot(4,1,4); hold on; grid on;
plot(selectedFrames, quat_A(selectedFrames,4), 'm-', 'LineWidth', 1.7);
plot(selectedFrames, quat_B(selectedFrames,4), 'm--', 'LineWidth', 1.7);
xlabel('Frame', 'FontSize', 14);
ylabel('q_z', 'FontSize', 14);
legend('Alg A q_z','Alg B q_z','Location','best');

sgtitle('Quaternion Comparison: Algorithm A vs Algorithm B', 'FontSize', 18);

print(gcf, fullfile(saveFolder, [comparisonName '_QuaternionComparison.png']), '-dpng','-r300');

%% (8) Compute differences between Algorithm A and B data
pos_diff = t_tgt_in_cam_A - t_tgt_in_cam_B;
eul_diff = eul_A - eul_B;
deg_diff = rad2deg(eul_diff);

% Restrict to selected frames for plotting
pos_diff_plot = pos_diff(selectedFrames,:);
deg_diff_plot = deg_diff(selectedFrames,:);

%% (9) Plot position differences (Algorithm A - Algorithm B)
figure('Name', 'Position Difference (Algorithm A - Algorithm B)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
subplot(3,1,1); hold on; grid on;
plot(selectedFrames, pos_diff_plot(:,1), 'r-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Diff X (m)');
title('Position Difference in X (Algorithm A - Algorithm B)');
subplot(3,1,2); hold on; grid on;
plot(selectedFrames, pos_diff_plot(:,2), 'g-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Diff Y (m)');
title('Position Difference in Y (Algorithm A - Algorithm B)');
subplot(3,1,3); hold on; grid on;
plot(selectedFrames, pos_diff_plot(:,3), 'b-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Diff Z (m)');
title('Position Difference in Z (Algorithm A - Algorithm B)');

print(gcf, fullfile(saveFolder, [comparisonName '_PositionDifference.png']), '-dpng', '-r300');

%% (10) Plot orientation differences (Algorithm A - Algorithm B)
figure('Name', 'Orientation Difference (Algorithm A - Algorithm B)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
subplot(3,1,1); hold on; grid on;
plot(selectedFrames, deg_diff_plot(:,1), 'r-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Roll Diff (deg)');
title('Orientation Difference in Roll (Algorithm A - Algorithm B)');
subplot(3,1,2); hold on; grid on;
plot(selectedFrames, deg_diff_plot(:,2), 'g-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Pitch Diff (deg)');
title('Orientation Difference in Pitch (Algorithm A - Algorithm B)');
subplot(3,1,3); hold on; grid on;
plot(selectedFrames, deg_diff_plot(:,3), 'b-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Yaw Diff (deg)');
title('Orientation Difference in Yaw (Algorithm A - Algorithm B)');

print(gcf, fullfile(saveFolder, [comparisonName '_OrientationDifference.png']), '-dpng', '-r300');

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
fprintf('\n--- Position Difference Statistics (Algorithm A - Algorithm B) in mm ---\n');
fprintf('Mean: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.mean(1), pos_stats.mean(2), pos_stats.mean(3));
fprintf('Std Dev: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.std(1), pos_stats.std(2), pos_stats.std(3));
fprintf('Max Abs: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.max(1), pos_stats.max(2), pos_stats.max(3));
fprintf('RMS: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.rms(1), pos_stats.rms(2), pos_stats.rms(3));

fprintf('\n--- Orientation Difference Statistics (Algorithm A - Algorithm B) in degrees ---\n');
fprintf('Mean: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.mean(1), ori_stats.mean(2), ori_stats.mean(3));
fprintf('Std Dev: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.std(1), ori_stats.std(2), ori_stats.std(3));
fprintf('Max Abs: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.max(1), ori_stats.max(2), ori_stats.max(3));
fprintf('RMS: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.rms(1), ori_stats.rms(2), ori_stats.rms(3));

%% (12) Calculate and plot distance measurements
% Calculate distances from camera to target for both algorithms
distances_A = vecnorm(t_tgt_in_cam_A, 2, 2);
distances_B = vecnorm(t_tgt_in_cam_B, 2, 2);

% Plot distance comparison
figure('Name', 'Distance Comparison (Algorithm A vs B)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 400]);
hold on; grid on;

plot(selectedFrames, distances_A(selectedFrames), 'b-', 'LineWidth', 1.5);
plot(selectedFrames, distances_B(selectedFrames), 'r-', 'LineWidth', 1.5);

% Plot difference
distance_diff = distances_A - distances_B;
plot(selectedFrames, distance_diff(selectedFrames), 'k-.', 'LineWidth', 1.5);

xlabel('Frame', 'FontSize', 14);
ylabel('Distance (m)', 'FontSize', 14);
title('Distance Comparison (Algorithm A vs Algorithm B)', 'FontSize', 16);
legend({'Algorithm A Distance', 'Algorithm B Distance', 'Difference (A - B)'}, 'Location', 'best');
hold off;

print(gcf, fullfile(saveFolder, [comparisonName '_DistanceComparison.png']), '-dpng', '-r300');

% Calculate distance statistics
dist_diff_mm = distance_diff * 1000; % Convert to mm
dist_stats = struct();
dist_stats.mean = mean(dist_diff_mm, 'omitnan');
dist_stats.std = std(dist_diff_mm, 'omitnan');
dist_stats.max = max(abs(dist_diff_mm), [], 'omitnan');
dist_stats.rms = sqrt(mean(dist_diff_mm.^2, 'omitnan'));

fprintf('\n--- Distance Difference Statistics (Algorithm A - Algorithm B) in mm ---\n');
fprintf('Mean: %.2f\n', dist_stats.mean);
fprintf('Std Dev: %.2f\n', dist_stats.std);
fprintf('Max Abs: %.2f\n', dist_stats.max);
fprintf('RMS: %.2f\n', dist_stats.rms);

%% (13) 3D Trajectory Comparison (Algorithm A vs B)
figure('Name', '3D Trajectory Comparison (Algorithm A vs B)', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 800]);
hold on; grid on; axis equal;
title('3D Trajectory: Algorithm A vs Algorithm B', 'FontSize', 18);

% Plot Algorithm A trajectory
plot3(t_tgt_in_cam_A(selectedFrames,1), ...
      t_tgt_in_cam_A(selectedFrames,2), ...
      t_tgt_in_cam_A(selectedFrames,3), ...
      'b-', 'LineWidth', 2);

% Plot Algorithm B trajectory
plot3(t_tgt_in_cam_B(selectedFrames,1), ...
      t_tgt_in_cam_B(selectedFrames,2), ...
      t_tgt_in_cam_B(selectedFrames,3), ...
      'r-', 'LineWidth', 2);

xlabel('X (m)', 'FontSize', 14);
ylabel('Y (m)', 'FontSize', 14);
zlabel('Z (m)', 'FontSize', 14);
legend({'Algorithm A Trajectory', 'Algorithm B Trajectory'}, 'Location', 'best');
view(3);  % 3D view

print(gcf, fullfile(saveFolder, [comparisonName '_3DTrajectoryComparison.png']), '-dpng', '-r300');

%% (14) 3D Coordinate Frame Visualization (Orientation + Position)
figure('Name', '3D Pose Visualization (Algorithm A vs B)', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 800]);
hold on; grid on; axis equal;
title('3D Poses: Coordinate Frames at Selected Steps', 'FontSize', 18);
xlabel('X (m)', 'FontSize', 14);
ylabel('Y (m)', 'FontSize', 14);
zlabel('Z (m)', 'FontSize', 14);

step = round(length(selectedFrames)/20);  % Show only a few for clarity

for idx = 1:step:length(selectedFrames)
    i = selectedFrames(idx);

    % Plot Algorithm A frames (solid lines)
    if ~isempty(R_tgt_in_cam_A{i}) && ~any(isnan(R_tgt_in_cam_A{i}(:))) && ~any(isnan(t_tgt_in_cam_A(i,:)))
        R = R_tgt_in_cam_A{i};
        t = t_tgt_in_cam_A(i,:)';
        scale = 0.1;  % Length of axes

        % X axis (Red)
        quiver3(t(1), t(2), t(3), R(1,1)*scale, R(2,1)*scale, R(3,1)*scale, 'r', 'LineWidth', 1.5);
        % Y axis (Green)
        quiver3(t(1), t(2), t(3), R(1,2)*scale, R(2,2)*scale, R(3,2)*scale, 'g', 'LineWidth', 1.5);
        % Z axis (Blue)
        quiver3(t(1), t(2), t(3), R(1,3)*scale, R(2,3)*scale, R(3,3)*scale, 'b', 'LineWidth', 1.5);
    end

    % Plot Algorithm B frames (dashed lines)
    if ~isempty(R_tgt_in_cam_B{i}) && ~any(isnan(R_tgt_in_cam_B{i}(:))) && ~any(isnan(t_tgt_in_cam_B(i,:)))
        R = R_tgt_in_cam_B{i};
        t = t_tgt_in_cam_B(i,:)';
        scale = 0.1;  % Length of axes

        % X axis (Red, dashed)
        quiver3(t(1), t(2), t(3), R(1,1)*scale, R(2,1)*scale, R(3,1)*scale, 'r--', 'LineWidth', 1.2);
        % Y axis (Green, dashed)
        quiver3(t(1), t(2), t(3), R(1,2)*scale, R(2,2)*scale, R(3,2)*scale, 'g--', 'LineWidth', 1.2);
        % Z axis (Blue, dashed)
        quiver3(t(1), t(2), t(3), R(1,3)*scale, R(2,3)*scale, R(3,3)*scale, 'b--', 'LineWidth', 1.2);
    end
end

legend({'Alg A X axis', 'Alg A Y axis', 'Alg A Z axis', 'Alg B X axis', 'Alg B Y axis', 'Alg B Z axis'}, 'Location', 'best');
view(3);

print(gcf, fullfile(saveFolder, [comparisonName '_3DPoseFrames.png']), '-dpng', '-r300');

fprintf('\n=== COMPARISON SUMMARY ===\n');
fprintf('Algorithm A: %s\n', jsonBase_A);
fprintf('Algorithm B: %s\n', jsonBase_B);
fprintf('Frames compared: %d\n', length(selectedFrames));
fprintf('Results saved in: %s\n', saveFolder);

disp("Finished plotting algorithm comparison. All figures saved automatically in: " + saveFolder);