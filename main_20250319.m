
%% Script to compare raw and refined/KF data from the same JSON file
% No longer using ROS bag (db3) file for ground truth

clear all; close all; clc;

%% (1) Preparations

% Edit this path to match your data:
jsonPath = 'KF_Upgrade.json';
%jsonPath = 'pose_estimation_MEKF_20250320_6.json';
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

validRawEntries = 0;
validKfEntries = 0;

for i = 1:N_json
    thisEntry = algData{i};

    % Store timestamp if available
    if isfield(thisEntry, 'timestamp')
        jsonTimes(i) = thisEntry.timestamp;
    end

    % Process raw data if available
    hasRawData = isfield(thisEntry, rawRotField) && isfield(thisEntry, rawTransField);
    if hasRawData
        validRawEntries = validRawEntries + 1;

        R_raw = thisEntry.(rawRotField);
        t_raw = thisEntry.(rawTransField);

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
    else
        R_tgt_in_cam_raw{i} = NaN(3,3);
    end

    % Process KF data if available
    hasKfData = isfield(thisEntry, kfRotField) && isfield(thisEntry, kfTransField);
    if hasKfData
        validKfEntries = validKfEntries + 1;

        R_kf = thisEntry.(kfRotField);
        t_kf = thisEntry.(kfTransField);

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
    else
        R_tgt_in_cam_kf{i} = NaN(3,3);
    end
end

fprintf("Found %d entries with raw data and %d entries with KF data\n", validRawEntries, validKfEntries);

%% (5) Compute Euler angles for both data types
eul_raw = zeros(N_json, 3);
eul_kf = zeros(N_json, 3);

for i = 1:N_json
    % Process raw rotation data if available
    if ~isempty(R_tgt_in_cam_raw{i}) && ~all(isnan(R_tgt_in_cam_raw{i}(:)))
        eul_raw(i,:) = rotm2eul(R_tgt_in_cam_raw{i}, 'XYZ');
    else
        eul_raw(i,:) = [NaN, NaN, NaN];
    end

    % Process KF rotation data if available
    if ~isempty(R_tgt_in_cam_kf{i}) && ~all(isnan(R_tgt_in_cam_kf{i}(:)))
        eul_kf(i,:) = rotm2eul(R_tgt_in_cam_kf{i}, 'XYZ');
    else
        eul_kf(i,:) = [NaN, NaN, NaN];
    end
end


%% (5.1) Compute quaternions for both data types
quat_raw = nan(N_json, 4);
quat_kf  = nan(N_json, 4);

for i = 1:N_json
    % Raw rotation to quaternion
    if ~isempty(R_tgt_in_cam_raw{i}) && ~all(isnan(R_tgt_in_cam_raw{i}(:)))
        quat_raw(i,:) = rotm2quat(R_tgt_in_cam_raw{i});  % returns [w x y z]
    end

    % KF rotation to quaternion
    if ~isempty(R_tgt_in_cam_kf{i}) && ~all(isnan(R_tgt_in_cam_kf{i}(:)))
        quat_kf(i,:) = rotm2quat(R_tgt_in_cam_kf{i});    % returns [w x y z]
    end
end


%% (6) Plot X/Y/Z position comparison for selected frames
figure('Name', 'Position Comparison (Raw vs KF)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
hold on; grid on;

% Create legend items dynamically based on available data
legendItems = {};

% Plot raw data if it exists
rawX = t_tgt_in_cam_raw(selectedFrames,1);
rawY = t_tgt_in_cam_raw(selectedFrames,2);
rawZ = t_tgt_in_cam_raw(selectedFrames,3);

if any(~isnan(rawX))
    plot(selectedFrames, rawX, 'b--', 'LineWidth', 1.7);
    legendItems = [legendItems, {'Raw X'}];
end

% Plot KF X data
kfX = t_tgt_in_cam_kf(selectedFrames,1);
if any(~isnan(kfX))
    plot(selectedFrames, kfX, 'b-', 'LineWidth', 1.7);
    legendItems = [legendItems, {'KF X'}];
end

% Plot raw Y data if it exists
if any(~isnan(rawY))
    plot(selectedFrames, rawY, 'g--', 'LineWidth', 1.7);
    legendItems = [legendItems, {'Raw Y'}];
end

% Plot KF Y data
kfY = t_tgt_in_cam_kf(selectedFrames,2);
if any(~isnan(kfY))
    plot(selectedFrames, kfY, 'g-', 'LineWidth', 1.7);
    legendItems = [legendItems, {'KF Y'}];
end

% Plot raw Z data if it exists
if any(~isnan(rawZ))
    plot(selectedFrames, rawZ, 'r--', 'LineWidth', 1.7);
    legendItems = [legendItems, {'Raw Z'}];
end

% Plot KF Z data
kfZ = t_tgt_in_cam_kf(selectedFrames,3);
if any(~isnan(kfZ))
    plot(selectedFrames, kfZ, 'r-', 'LineWidth', 1.7);
    legendItems = [legendItems, {'KF Z'}];
end

set(gca, 'FontSize', 14);
xlabel('Frame', 'FontSize', 16); 
ylabel('Position (m)', 'FontSize', 16);
legend(legendItems, 'Location', 'best');
title('Position Comparison: Raw vs KF', 'FontSize', 18);
hold off;

print(gcf, fullfile(saveFolder, ...
    [jsonBase '_PositionComparison.png']), '-dpng', '-r300');

%% (7) Orientation comparison for selected frames (Euler angles)
figure('Name', 'Orientation Comparison (Raw vs KF)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
hold on; grid on;

% Create legend items dynamically based on available data
legendItems = {};

% Plot raw orientation data if it exists
rawRoll = rad2deg(eul_raw(selectedFrames,1));
rawPitch = rad2deg(eul_raw(selectedFrames,2));
rawYaw = rad2deg(eul_raw(selectedFrames,3));

if any(~isnan(rawRoll))
    plot(selectedFrames, rawRoll, 'b--', 'LineWidth', 1.7);
    legendItems = [legendItems, {'Raw Roll'}];
end

% Plot KF roll data
kfRoll = rad2deg(eul_kf(selectedFrames,1));
if any(~isnan(kfRoll))
    plot(selectedFrames, kfRoll, 'b-', 'LineWidth', 1.7);
    legendItems = [legendItems, {'KF Roll'}];
end

% Plot raw pitch data if it exists
if any(~isnan(rawPitch))
    plot(selectedFrames, rawPitch, 'g--', 'LineWidth', 1.7);
    legendItems = [legendItems, {'Raw Pitch'}];
end

% Plot KF pitch data
kfPitch = rad2deg(eul_kf(selectedFrames,2));
if any(~isnan(kfPitch))
    plot(selectedFrames, kfPitch, 'g-', 'LineWidth', 1.7);
    legendItems = [legendItems, {'KF Pitch'}];
end

% Plot raw yaw data if it exists
if any(~isnan(rawYaw))
    plot(selectedFrames, rawYaw, 'm--', 'LineWidth', 1.7);
    legendItems = [legendItems, {'Raw Yaw'}];
end

% Plot KF yaw data
kfYaw = rad2deg(eul_kf(selectedFrames,3));
if any(~isnan(kfYaw))
    plot(selectedFrames, kfYaw, 'm-', 'LineWidth', 1.7);
    legendItems = [legendItems, {'KF Yaw'}];
end

set(gca, 'FontSize', 14);
xlabel('Frame', 'FontSize', 16); 
ylabel('Angle (deg)', 'FontSize', 16);
legend(legendItems, 'Location', 'best');
title('Orientation Comparison: Raw vs KF', 'FontSize', 18);
hold off;

print(gcf, fullfile(saveFolder, ...
    [jsonBase '_OrientationComparison.png']), '-dpng', '-r300');



%% (7.1) Plot Quaternion Comparison (Raw vs KF)
figure('Name','Quaternion Comparison (Raw vs KF)', ...
       'NumberTitle','off', 'Position',[100, 100, 900, 800]);

% w-component
subplot(4,1,1); hold on; grid on;
plot(selectedFrames, quat_raw(selectedFrames,1), 'b--', 'LineWidth', 1.7);
plot(selectedFrames, quat_kf(selectedFrames,1),  'b-',  'LineWidth', 1.7);
ylabel('q_w', 'FontSize', 14);
legend('Raw q_w','KF q_w','Location','best');

% x-component
subplot(4,1,2); hold on; grid on;
plot(selectedFrames, quat_raw(selectedFrames,2), 'g--', 'LineWidth', 1.7);
plot(selectedFrames, quat_kf(selectedFrames,2),  'g-',  'LineWidth', 1.7);
ylabel('q_x', 'FontSize', 14);
legend('Raw q_x','KF q_x','Location','best');

% y-component
subplot(4,1,3); hold on; grid on;
plot(selectedFrames, quat_raw(selectedFrames,3), 'r--', 'LineWidth', 1.7);
plot(selectedFrames, quat_kf(selectedFrames,3),  'r-',  'LineWidth', 1.7);
ylabel('q_y', 'FontSize', 14);
legend('Raw q_y','KF q_y','Location','best');

% z-component
subplot(4,1,4); hold on; grid on;
plot(selectedFrames, quat_raw(selectedFrames,4), 'm--', 'LineWidth', 1.7);
plot(selectedFrames, quat_kf(selectedFrames,4),  'm-',  'LineWidth', 1.7);
xlabel('Frame', 'FontSize', 14);
ylabel('q_z', 'FontSize', 14);
legend('Raw q_z','KF q_z','Location','best');

sgtitle('Quaternion Comparison: Raw vs KF', 'FontSize', 18);

% Optionally save the figure
print(gcf, fullfile(saveFolder, [jsonBase '_QuaternionComparison.png']), '-dpng','-r300');




% %% (8) Compute differences between raw and KF data (only where both exist)
% pos_diff = t_tgt_in_cam_raw - t_tgt_in_cam_kf;
% eul_diff = eul_raw - eul_kf;
% deg_diff = rad2deg(eul_diff);
% 
% % Restrict to selected frames for plotting
% pos_diff_plot = pos_diff(selectedFrames,:);
% deg_diff_plot = deg_diff(selectedFrames,:);
% 
% %% (9) Plot position differences (only if raw data exists)
% % Only create the difference plots if there's raw data to compare with
% if validRawEntries > 0
%     figure('Name', 'Position Difference (Raw - KF)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
%     subplot(3,1,1); hold on; grid on;
%     plot(selectedFrames, pos_diff_plot(:,1), 'r-', 'LineWidth', 1.5);
%     xlabel('Frame'); ylabel('Diff X (m)');
%     title('Position Difference in X (Raw - KF)');
%     subplot(3,1,2); hold on; grid on;
%     plot(selectedFrames, pos_diff_plot(:,2), 'g-', 'LineWidth', 1.5);
%     xlabel('Frame'); ylabel('Diff Y (m)');
%     title('Position Difference in Y (Raw - KF)');
%     subplot(3,1,3); hold on; grid on;
%     plot(selectedFrames, pos_diff_plot(:,3), 'b-', 'LineWidth', 1.5);
%     xlabel('Frame'); ylabel('Diff Z (m)');
%     title('Position Difference in Z (Raw - KF)');
% 
%     print(gcf, fullfile(saveFolder, ...
%         [jsonBase '_PositionDifference.png']), '-dpng', '-r300');
% 
%     %% (10) Plot orientation differences (only if raw data exists)
%     figure('Name', 'Orientation Difference (Raw - KF)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);
%     subplot(3,1,1); hold on; grid on;
%     plot(selectedFrames, deg_diff_plot(:,1), 'r-', 'LineWidth', 1.5);
%     xlabel('Frame'); ylabel('Roll Diff (deg)');
%     title('Orientation Difference in Roll (Raw - KF)');
%     subplot(3,1,2); hold on; grid on;
%     plot(selectedFrames, deg_diff_plot(:,2), 'g-', 'LineWidth', 1.5);
%     xlabel('Frame'); ylabel('Pitch Diff (deg)');
%     title('Orientation Difference in Pitch (Raw - KF)');
%     subplot(3,1,3); hold on; grid on;
%     plot(selectedFrames, deg_diff_plot(:,3), 'b-', 'LineWidth', 1.5);
%     xlabel('Frame'); ylabel('Yaw Diff (deg)');
%     title('Orientation Difference in Yaw (Raw - KF)');
% 
%     print(gcf, fullfile(saveFolder, ...
%         [jsonBase '_OrientationDifference.png']), '-dpng', '-r300');
% 
%     %% (11) Calculate and display statistics (only where both raw and KF data exist)
%     % Calculate statistics for position differences (in mm for better readability)
%     pos_diff_mm = pos_diff * 1000; % Convert to mm
%     pos_stats = struct();
%     pos_stats.mean = mean(pos_diff_mm, 'omitnan');
%     pos_stats.std = std(pos_diff_mm, 'omitnan');
%     pos_stats.max = max(abs(pos_diff_mm), [], 'omitnan');
%     pos_stats.rms = sqrt(mean(pos_diff_mm.^2, 'omitnan'));
% 
%     % Calculate statistics for orientation differences (in degrees)
%     ori_stats = struct();
%     ori_stats.mean = mean(deg_diff, 'omitnan');
%     ori_stats.std = std(deg_diff, 'omitnan');
%     ori_stats.max = max(abs(deg_diff), [], 'omitnan');
%     ori_stats.rms = sqrt(mean(deg_diff.^2, 'omitnan'));
% 
%     % Display statistics
%     fprintf('\n--- Position Difference Statistics (Raw - KF) in mm ---\n');
%     fprintf('Mean: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.mean(1), pos_stats.mean(2), pos_stats.mean(3));
%     fprintf('Std Dev: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.std(1), pos_stats.std(2), pos_stats.std(3));
%     fprintf('Max Abs: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.max(1), pos_stats.max(2), pos_stats.max(3));
%     fprintf('RMS: X=%.2f, Y=%.2f, Z=%.2f\n', pos_stats.rms(1), pos_stats.rms(2), pos_stats.rms(3));
% 
%     fprintf('\n--- Orientation Difference Statistics (Raw - KF) in degrees ---\n');
%     fprintf('Mean: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.mean(1), ori_stats.mean(2), ori_stats.mean(3));
%     fprintf('Std Dev: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.std(1), ori_stats.std(2), ori_stats.std(3));
%     fprintf('Max Abs: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.max(1), ori_stats.max(2), ori_stats.max(3));
%     fprintf('RMS: Roll=%.2f, Pitch=%.2f, Yaw=%.2f\n', ori_stats.rms(1), ori_stats.rms(2), ori_stats.rms(3));
% end
% 
% %% (12) Optional: Calculate and plot distance measurements
% % Calculate distances from camera to target for both raw and KF data
% raw_distances = vecnorm(t_tgt_in_cam_raw, 2, 2);
% kf_distances = vecnorm(t_tgt_in_cam_kf, 2, 2);
% 
% % Plot distance comparison, always showing KF data
% figure('Name', 'Distance Comparison (Raw vs KF)', 'NumberTitle', 'off', 'Position', [100, 100, 900, 400]);
% hold on; grid on;
% legendItems = {};
% 
% % Plot raw distances if they exist
% if any(~isnan(raw_distances))
%     plot(selectedFrames, raw_distances(selectedFrames), 'b--', 'LineWidth', 1.5);
%     legendItems = [legendItems, {'Raw Distance'}];
% end
% 
% % Plot KF distances
% if any(~isnan(kf_distances))
%     plot(selectedFrames, kf_distances(selectedFrames), 'r-', 'LineWidth', 1.5);
%     legendItems = [legendItems, {'KF Distance'}];
% end
% 
% % Plot difference if both exist
% if any(~isnan(raw_distances)) && any(~isnan(kf_distances))
%     distance_diff = raw_distances - kf_distances;
%     plot(selectedFrames, distance_diff(selectedFrames), 'k-.', 'LineWidth', 1.5);
%     legendItems = [legendItems, {'Difference (Raw - KF)'}];
% end
% 
% xlabel('Frame', 'FontSize', 14);
% ylabel('Distance (m)', 'FontSize', 14);
% title('Distance Comparison (Raw vs KF)', 'FontSize', 16);
% if ~isempty(legendItems)
%     legend(legendItems, 'Location', 'best');
% end
% hold off;
% 
% print(gcf, fullfile(saveFolder, ...
%     [jsonBase '_DistanceComparison.png']), '-dpng', '-r300');
% 
% % Calculate distance statistics if both raw and KF data exist
% if validRawEntries > 0 && validKfEntries > 0
%     distance_diff = raw_distances - kf_distances;
%     dist_diff_mm = distance_diff * 1000; % Convert to mm
%     dist_stats = struct();
%     dist_stats.mean = mean(dist_diff_mm, 'omitnan');
%     dist_stats.std = std(dist_diff_mm, 'omitnan');
%     dist_stats.max = max(abs(dist_diff_mm), [], 'omitnan');
%     dist_stats.rms = sqrt(mean(dist_diff_mm.^2, 'omitnan'));
% 
%     fprintf('\n--- Distance Difference Statistics (Raw - KF) in mm ---\n');
%     fprintf('Mean: %.2f\n', dist_stats.mean);
%     fprintf('Std Dev: %.2f\n', dist_stats.std);
%     fprintf('Max Abs: %.2f\n', dist_stats.max);
%     fprintf('RMS: %.2f\n', dist_stats.rms);
% end

disp("Finished plotting. All figures saved automatically in: " + saveFolder);