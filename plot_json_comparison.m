%% Script to compare Kalman Filtered (KF) data from two different JSON files
% Each JSON file is assumed to be the result of a different pose estimation algorithm
% for the same sequence of events.

clear all; close all; clc;

%% (1) Preparations

% --- EDIT THE FOLLOWING PATHS ---
% Path to the first JSON file (e.g., Algorithm 1)
jsonPath1 = 'VAPE_outdoor.json'; 
% Path to the second JSON file (e.g., Algorithm 2)
jsonPath2 = 'RTM_outdoor.json'; 

% --- Optional: Names for the legend ---
algoName1 = 'VAPE';
algoName2 = 'RTMpose';
% --------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% Create a "Matlab_results" folder and a subfolder with today's date to save plots
baseFolder = 'Matlab_results';
if ~exist(baseFolder, 'dir')
    mkdir(baseFolder);
end

dateStr = datestr(now,'yyyy-mm-dd');

% Create a descriptive folder name for the comparison
[~, jsonBase1, ~] = fileparts(jsonPath1);
[~, jsonBase2, ~] = fileparts(jsonPath2);
comparisonName = [jsonBase1 '_vs_' jsonBase2];
testCaseFolder = fullfile(baseFolder, dateStr, comparisonName);

if ~exist(testCaseFolder, 'dir')
    mkdir(testCaseFolder);
end

saveFolder = testCaseFolder;

%% (2) Load Data from Both JSON Files

% --- Load First JSON File ---
fprintf('Loading data from: %s\n', jsonPath1);
jsonText1 = fileread(jsonPath1);
algData1 = jsondecode(jsonText1);
N1 = numel(algData1);
fprintf('Loaded %d frames from %s\n', N1, jsonPath1);

% --- Load Second JSON File ---
fprintf('Loading data from: %s\n', jsonPath2);
jsonText2 = fileread(jsonPath2);
algData2 = jsondecode(jsonText2);
N2 = numel(algData2);
fprintf('Loaded %d frames from %s\n', N2, jsonPath2);

%% (3) Frame selection for plotting
% Use the minimum number of frames available in the two files for a fair comparison
N_json = min(N1, N2);

% Specify the desired number of frames to plot
desiredFrameCount = 560;
% Automatically select frames from 1 up to the lesser of desiredFrameCount or the available frames
selectedFrames = 1:min(desiredFrameCount, N_json);
if max(selectedFrames) > N_json
    error('Selected frames exceed the available frames in one of the JSON files.');
end
fprintf('Comparing %d frames.\n', length(selectedFrames));

%% (4) Extract KF data from both JSON files
% We'll store the rotation and translation from both algorithms
R_tgt_in_cam_kf1 = cell(N_json, 1);
t_tgt_in_cam_kf1 = nan(N_json, 3);
R_tgt_in_cam_kf2 = cell(N_json, 1);
t_tgt_in_cam_kf2 = nan(N_json, 3);
jsonTimes1 = nan(N_json, 1);
jsonTimes2 = nan(N_json, 1);

% Define field names for KF data (assuming they are the same in both files)
kfRotField = 'rotation_matrix';
kfTransField = 'position';

% --- Extract from First File ---
validKfEntries1 = 0;
for i = 1:N_json
    thisEntry = algData1{i};
    if isfield(thisEntry, kfRotField) && isfield(thisEntry, kfTransField)
        validKfEntries1 = validKfEntries1 + 1;
        R_kf = thisEntry.(kfRotField);
        t_kf = thisEntry.(kfTransField);
        if ~isempty(R_kf) && numel(R_kf) == 9
            R_tgt_in_cam_kf1{i} = reshape(R_kf, [3,3]);
        else
            R_tgt_in_cam_kf1{i} = NaN(3,3);
        end
        if ~isempty(t_kf) && numel(t_kf) == 3
            t_tgt_in_cam_kf1(i,:) = t_kf(:)';
        end
        if isfield(thisEntry, 'timestamp')
            jsonTimes1(i) = thisEntry.timestamp;
        end
    else
        R_tgt_in_cam_kf1{i} = NaN(3,3);
    end
end

% --- Extract from Second File ---
validKfEntries2 = 0;
for i = 1:N_json
    thisEntry = algData2{i};
    if isfield(thisEntry, kfRotField) && isfield(thisEntry, kfTransField)
        validKfEntries2 = validKfEntries2 + 1;
        R_kf = thisEntry.(kfRotField);
        t_kf = thisEntry.(kfTransField);
        if ~isempty(R_kf) && numel(R_kf) == 9
            R_tgt_in_cam_kf2{i} = reshape(R_kf, [3,3]);
        else
            R_tgt_in_cam_kf2{i} = NaN(3,3);
        end
        if ~isempty(t_kf) && numel(t_kf) == 3
            t_tgt_in_cam_kf2(i,:) = t_kf(:)';
        end
        if isfield(thisEntry, 'timestamp')
            jsonTimes2(i) = thisEntry.timestamp;
        end
    else
        R_tgt_in_cam_kf2{i} = NaN(3,3);
    end
end

fprintf("Found %d valid KF entries in %s and %d in %s\n", validKfEntries1, jsonPath1, validKfEntries2, jsonPath2);

%% (5) Compute Euler angles for both KF datasets
eul_kf1 = zeros(N_json, 3);
eul_kf2 = zeros(N_json, 3);

for i = 1:N_json
    if ~isempty(R_tgt_in_cam_kf1{i}) && ~all(isnan(R_tgt_in_cam_kf1{i}(:)))
        eul_kf1(i,:) = rotm2eul(R_tgt_in_cam_kf1{i}, 'XYZ');
    else
        eul_kf1(i,:) = [NaN, NaN, NaN];
    end
    if ~isempty(R_tgt_in_cam_kf2{i}) && ~all(isnan(R_tgt_in_cam_kf2{i}(:)))
        eul_kf2(i,:) = rotm2eul(R_tgt_in_cam_kf2{i}, 'XYZ');
    else
        eul_kf2(i,:) = [NaN, NaN, NaN];
    end
end

%% (5.1) Compute quaternions for both KF datasets
quat_kf1 = nan(N_json, 4);
quat_kf2 = nan(N_json, 4);

for i = 1:N_json
    if ~isempty(R_tgt_in_cam_kf1{i}) && ~all(isnan(R_tgt_in_cam_kf1{i}(:)))
        quat_kf1(i,:) = rotm2quat(R_tgt_in_cam_kf1{i});
    end
    if ~isempty(R_tgt_in_cam_kf2{i}) && ~all(isnan(R_tgt_in_cam_kf2{i}(:)))
        quat_kf2(i,:) = rotm2quat(R_tgt_in_cam_kf2{i});
    end
end

%% (6) Plot X/Y/Z position comparison for selected frames
figure('Name', 'Position Comparison (Algo1 vs Algo2)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
hold on; grid on;

plot(selectedFrames, t_tgt_in_cam_kf1(selectedFrames,1), 'b-', 'LineWidth', 1.7);
plot(selectedFrames, t_tgt_in_cam_kf2(selectedFrames,1), 'b--', 'LineWidth', 1.7);

plot(selectedFrames, t_tgt_in_cam_kf1(selectedFrames,2), 'g-', 'LineWidth', 1.7);
plot(selectedFrames, t_tgt_in_cam_kf2(selectedFrames,2), 'g--', 'LineWidth', 1.7);

plot(selectedFrames, t_tgt_in_cam_kf1(selectedFrames,3), 'r-', 'LineWidth', 1.7);
plot(selectedFrames, t_tgt_in_cam_kf2(selectedFrames,3), 'r--', 'LineWidth', 1.7);

set(gca, 'FontSize', 14);
xlabel('Frame', 'FontSize', 16); 
ylabel('Position (m)', 'FontSize', 16);
legend({[algoName1 ' X'], [algoName2 ' X'], ...
        [algoName1 ' Y'], [algoName2 ' Y'], ...
        [algoName1 ' Z'], [algoName2 ' Z']}, 'Location', 'best');
title(['Position Comparison: ' algoName1 ' vs ' algoName2], 'FontSize', 18);
hold off;

print(gcf, fullfile(saveFolder, [comparisonName '_PositionComparison.png']), '-dpng', '-r300');

%% (7) Orientation comparison for selected frames (Euler angles)
figure('Name', 'Orientation Comparison (Algo1 vs Algo2)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
hold on; grid on;

plot(selectedFrames, rad2deg(eul_kf1(selectedFrames,1)), 'b-', 'LineWidth', 1.7);
plot(selectedFrames, rad2deg(eul_kf2(selectedFrames,1)), 'b--', 'LineWidth', 1.7);

plot(selectedFrames, rad2deg(eul_kf1(selectedFrames,2)), 'g-', 'LineWidth', 1.7);
plot(selectedFrames, rad2deg(eul_kf2(selectedFrames,2)), 'g--', 'LineWidth', 1.7);

plot(selectedFrames, rad2deg(eul_kf1(selectedFrames,3)), 'm-', 'LineWidth', 1.7);
plot(selectedFrames, rad2deg(eul_kf2(selectedFrames,3)), 'm--', 'LineWidth', 1.7);

set(gca, 'FontSize', 14);
xlabel('Frame', 'FontSize', 16); 
ylabel('Angle (deg)', 'FontSize', 16);
legend({[algoName1 ' Roll'], [algoName2 ' Roll'], ...
        [algoName1 ' Pitch'], [algoName2 ' Pitch'], ...
        [algoName1 ' Yaw'], [algoName2 ' Yaw']}, 'Location', 'best');
title(['Orientation Comparison: ' algoName1 ' vs ' algoName2], 'FontSize', 18);
hold off;

print(gcf, fullfile(saveFolder, [comparisonName '_OrientationComparison.png']), '-dpng', '-r300');

%% (7.1) Plot Quaternion Comparison (Algo1 vs Algo2)
figure('Name','Quaternion Comparison (Algo1 vs Algo2)', ...
       'NumberTitle','off', 'Position',[100, 100, 900, 800]);

% w-component
subplot(4,1,1); hold on; grid on;
plot(selectedFrames, quat_kf1(selectedFrames,1), 'b-', 'LineWidth', 1.7);
plot(selectedFrames, quat_kf2(selectedFrames,1), 'b--', 'LineWidth', 1.7);
ylabel('q_w', 'FontSize', 14);
legend([algoName1 ' q_w'], [algoName2 ' q_w'],'Location','best');

% x-component
subplot(4,1,2); hold on; grid on;
plot(selectedFrames, quat_kf1(selectedFrames,2), 'g-', 'LineWidth', 1.7);
plot(selectedFrames, quat_kf2(selectedFrames,2), 'g--', 'LineWidth', 1.7);
ylabel('q_x', 'FontSize', 14);
legend([algoName1 ' q_x'], [algoName2 ' q_x'],'Location','best');

% y-component
subplot(4,1,3); hold on; grid on;
plot(selectedFrames, quat_kf1(selectedFrames,3), 'r-', 'LineWidth', 1.7);
plot(selectedFrames, quat_kf2(selectedFrames,3), 'r--', 'LineWidth', 1.7);
ylabel('q_y', 'FontSize', 14);
legend([algoName1 ' q_y'], [algoName2 ' q_y'],'Location','best');

% z-component
subplot(4,1,4); hold on; grid on;
plot(selectedFrames, quat_kf1(selectedFrames,4), 'm-', 'LineWidth', 1.7);
plot(selectedFrames, quat_kf2(selectedFrames,4), 'm--', 'LineWidth', 1.7);
xlabel('Frame', 'FontSize', 14);
ylabel('q_z', 'FontSize', 14);
legend([algoName1 ' q_z'], [algoName2 ' q_z'],'Location','best');

sgtitle(['Quaternion Comparison: ' algoName1 ' vs ' algoName2], 'FontSize', 18);

print(gcf, fullfile(saveFolder, [comparisonName '_QuaternionComparison.png']), '-dpng','-r300');

%% (8) Compute differences between the two KF datasets
pos_diff = t_tgt_in_cam_kf1 - t_tgt_in_cam_kf2;
eul_diff = eul_kf1 - eul_kf2;
% Handle angle wrapping for yaw/roll/pitch differences
eul_diff = atan2(sin(eul_diff), cos(eul_diff));
deg_diff = rad2deg(eul_diff);

% Restrict to selected frames for plotting
pos_diff_plot = pos_diff(selectedFrames,:);
deg_diff_plot = deg_diff(selectedFrames,:);

%% (9) Plot position differences
figure('Name', 'Position Difference (Algo1 - Algo2)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
subplot(3,1,1); hold on; grid on;
plot(selectedFrames, pos_diff_plot(:,1), 'r-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Diff X (m)');
title(['Position Difference in X (' algoName1 ' - ' algoName2 ')']);
subplot(3,1,2); hold on; grid on;
plot(selectedFrames, pos_diff_plot(:,2), 'g-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Diff Y (m)');
title(['Position Difference in Y (' algoName1 ' - ' algoName2 ')']);
subplot(3,1,3); hold on; grid on;
plot(selectedFrames, pos_diff_plot(:,3), 'b-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Diff Z (m)');
title(['Position Difference in Z (' algoName1 ' - ' algoName2 ')']);

print(gcf, fullfile(saveFolder, [comparisonName '_PositionDifference.png']), '-dpng', '-r300');

%% (10) Plot orientation differences
figure('Name', 'Orientation Difference (Algo1 - Algo2)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
subplot(3,1,1); hold on; grid on;
plot(selectedFrames, deg_diff_plot(:,1), 'r-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Roll Diff (deg)');
title(['Orientation Difference in Roll (' algoName1 ' - ' algoName2 ')']);
subplot(3,1,2); hold on; grid on;
plot(selectedFrames, deg_diff_plot(:,2), 'g-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Pitch Diff (deg)');
title(['Orientation Difference in Pitch (' algoName1 ' - ' algoName2 ')']);
subplot(3,1,3); hold on; grid on;
plot(selectedFrames, deg_diff_plot(:,3), 'b-', 'LineWidth', 1.5);
xlabel('Frame'); ylabel('Yaw Diff (deg)');
title(['Orientation Difference in Yaw (' algoName1 ' - ' algoName2 ')']);

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
fprintf('\n--- Difference Statistics (%s - %s) ---\n', algoName1, algoName2);
fprintf('--- Position Difference (mm) ---\n');
fprintf('Mean: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.mean(1), pos_stats.mean(2), pos_stats.mean(3));
fprintf('Std Dev: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.std(1), pos_stats.std(2), pos_stats.std(3));
fprintf('Max Abs: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.max(1), pos_stats.max(2), pos_stats.max(3));
fprintf('RMS: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.rms(1), pos_stats.rms(2), pos_stats.rms(3));

fprintf('\n--- Orientation Difference (degrees) ---\n');
fprintf('Mean: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.mean(1), ori_stats.mean(2), ori_stats.mean(3));
fprintf('Std Dev: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.std(1), ori_stats.std(2), ori_stats.std(3));
fprintf('Max Abs: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.max(1), ori_stats.max(2), ori_stats.max(3));
fprintf('RMS: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.rms(1), ori_stats.rms(2), ori_stats.rms(3));

%% (12) Optional: Calculate and plot distance measurements
% Calculate distances from camera to target for both KF datasets
kf1_distances = vecnorm(t_tgt_in_cam_kf1, 2, 2);
kf2_distances = vecnorm(t_tgt_in_cam_kf2, 2, 2);

% Plot distance comparison
figure('Name', 'Distance Comparison (Algo1 vs Algo2)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 400]);
hold on; grid on;

plot(selectedFrames, kf1_distances(selectedFrames), 'r-', 'LineWidth', 1.5);
plot(selectedFrames, kf2_distances(selectedFrames), 'b--', 'LineWidth', 1.5);

distance_diff = kf1_distances - kf2_distances;
plot(selectedFrames, distance_diff(selectedFrames), 'k-.', 'LineWidth', 1.5);

xlabel('Frame', 'FontSize', 14);
ylabel('Distance (m)', 'FontSize', 14);
title(['Distance Comparison: ' algoName1 ' vs ' algoName2], 'FontSize', 16);
legend({[algoName1 ' Distance'], [algoName2 ' Distance'], 'Difference'}, 'Location', 'best');
hold off;

print(gcf, fullfile(saveFolder, [comparisonName '_DistanceComparison.png']), '-dpng', '-r300');

% Calculate distance statistics
distance_diff_mm = distance_diff * 1000; % Convert to mm
dist_stats = struct();
dist_stats.mean = mean(distance_diff_mm, 'omitnan');
dist_stats.std = std(distance_diff_mm, 'omitnan');
dist_stats.max = max(abs(distance_diff_mm), [], 'omitnan');
dist_stats.rms = sqrt(mean(distance_diff_mm.^2, 'omitnan'));

fprintf('\n--- Distance Difference Statistics (%s - %s) in mm ---\n', algoName1, algoName2);
fprintf('Mean: %.2f\n', dist_stats.mean);
fprintf('Std Dev: %.2f\n', dist_stats.std);
fprintf('Max Abs: %.2f\n', dist_stats.max);
fprintf('RMS: %.2f\n', dist_stats.rms);

%% (13) 3D Trajectory Comparison
figure('Name', '3D Trajectory Comparison (Algo1 vs Algo2)', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 800]);
hold on; grid on; axis equal;
title(['3D Trajectory: ' algoName1 ' vs ' algoName2], 'FontSize', 18);

% Plot Algo 1 trajectory
plot3(t_tgt_in_cam_kf1(selectedFrames,1), ...
      t_tgt_in_cam_kf1(selectedFrames,2), ...
      t_tgt_in_cam_kf1(selectedFrames,3), ...
      'r-', 'LineWidth', 2);

% Plot Algo 2 trajectory
plot3(t_tgt_in_cam_kf2(selectedFrames,1), ...
      t_tgt_in_cam_kf2(selectedFrames,2), ...
      t_tgt_in_cam_kf2(selectedFrames,3), ...
      'b--', 'LineWidth', 2);

xlabel('X (m)', 'FontSize', 14);
ylabel('Y (m)', 'FontSize', 14);
zlabel('Z (m)', 'FontSize', 14);
legend({[algoName1 ' Trajectory'], [algoName2 ' Trajectory']}, 'Location', 'best');
view(3);

print(gcf, fullfile(saveFolder, [comparisonName '_3DTrajectoryComparison.png']), '-dpng', '-r300');

%% (14) 3D Coordinate Frame Visualization (Orientation + Position)
figure('Name', '3D Pose Visualization (Algo1 vs Algo2)', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 800]);
hold on; grid on; axis equal;
title('3D Poses: Coordinate Frames at Selected Steps', 'FontSize', 18);
xlabel('X (m)', 'FontSize', 14);
ylabel('Y (m)', 'FontSize', 14);
zlabel('Z (m)', 'FontSize', 14);

step = round(length(selectedFrames)/10); % Show fewer frames for clarity

for idx = 1:step:length(selectedFrames)
    i = selectedFrames(idx);

    % Plot Algo 1 frames (in red-green-blue for x-y-z)
    if ~isempty(R_tgt_in_cam_kf1{i}) && ~any(isnan(R_tgt_in_cam_kf1{i}(:))) && ~any(isnan(t_tgt_in_cam_kf1(i,:)))
        R = R_tgt_in_cam_kf1{i};
        t = t_tgt_in_cam_kf1(i,:)';
        scale = 0.1;
        quiver3(t(1), t(2), t(3), R(1,1)*scale, R(2,1)*scale, R(3,1)*scale, 'r', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        quiver3(t(1), t(2), t(3), R(1,2)*scale, R(2,2)*scale, R(3,2)*scale, 'g', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        quiver3(t(1), t(2), t(3), R(1,3)*scale, R(2,3)*scale, R(3,3)*scale, 'b', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    end
    
    % Plot Algo 2 frames (in cyan-magenta-yellow for x-y-z)
    if ~isempty(R_tgt_in_cam_kf2{i}) && ~any(isnan(R_tgt_in_cam_kf2{i}(:))) && ~any(isnan(t_tgt_in_cam_kf2(i,:)))
        R = R_tgt_in_cam_kf2{i};
        t = t_tgt_in_cam_kf2(i,:)';
        scale = 0.1;
        quiver3(t(1), t(2), t(3), R(1,1)*scale, R(2,1)*scale, R(3,1)*scale, 'c', 'LineWidth', 1.2, 'LineStyle', '--', 'HandleVisibility', 'off');
        quiver3(t(1), t(2), t(3), R(1,2)*scale, R(2,2)*scale, R(3,2)*scale, 'm', 'LineWidth', 1.2, 'LineStyle', '--', 'HandleVisibility', 'off');
        quiver3(t(1), t(2), t(3), R(1,3)*scale, R(2,3)*scale, R(3,3)*scale, 'y', 'LineWidth', 1.2, 'LineStyle', '--', 'HandleVisibility', 'off');
    end
end

% Create dummy plots for the legend
h1 = plot(NaN,NaN,'r-');
h2 = plot(NaN,NaN,'b-');
legend([h1, h2], {algoName1, algoName2}, 'Location', 'best');
view(3);

print(gcf, fullfile(saveFolder, [comparisonName '_3DPoseFrames.png']), '-dpng', '-r300');

disp("Finished plotting. All figures saved automatically in: " + saveFolder);
