%% (0) Preparations
clear all;
close all;
clc;

% Example total duration in seconds (if known or assumed):
T = 22;  % e.g., 25 seconds of data

jsonPath   = '20250108_test2_pnp_thresh_RE.json';%'20250108_thresh_test18.json';   % JSON file (estimated poses)
folderPath = '20250107_test2.db3';               % ROS2 bag (.db3) file (GT data)

%% (A) Load ground truth (OptiTrack) from ROS2 bag
bagReader  = ros2bagreader(folderPath);
msgs       = readMessages(bagReader);
N_opt      = length(msgs);  % Total frames in bag (e.g., 218)

cam_pos_world  = zeros(N_opt,3);
cam_quat_world = zeros(N_opt,4);
tgt_pos_world  = zeros(N_opt,3);
tgt_quat_world = zeros(N_opt,4);

for i = 1:N_opt
    d = msgs{i}.data;
    cam_pos_world(i,:)  = d(1:3);
    cam_quat_world(i,:) = d(4:7);
    tgt_pos_world(i,:)  = d(8:10);
    tgt_quat_world(i,:) = d(11:14);
end

% Convert quaternions to rotation matrices
R_cam_in_world = zeros(3,3,N_opt);
R_tgt_in_world = zeros(3,3,N_opt);
for i = 1:N_opt
    R_cam_in_world(:,:,i) = quat2rotm(cam_quat_world(i,:));
    R_tgt_in_world(:,:,i) = quat2rotm(tgt_quat_world(i,:));
end

%% (B) Load the algorithm’s estimated data from JSON
jsonText = fileread(jsonPath);
algData  = jsondecode(jsonText);

% Extract frame indices and pose data
present_frames = arrayfun(@(x) x.frame, algData);  % Extract available frames
N_json = max(present_frames);  % Total expected frames (based on the maximum frame index)

R_tgt_in_cam_alg = cell(N_json,1);
t_tgt_in_cam_alg = nan(N_json,3);  % Initialize with NaN for missing frames

% Populate estimated data
for i = 1:length(algData)
    frame_idx = algData(i).frame;
    R_arr = algData(i).kf_rotation_matrix;
    t_arr = algData(i).kf_translation_vector;
    %R_arr = algData(i).object_rotation_in_cam;
    %t_arr = algData(i).object_translation_in_cam;

    R_tgt_in_cam_alg{frame_idx} = reshape(R_arr, [3,3]);
    t_tgt_in_cam_alg(frame_idx,:) = t_arr;
end

% Identify missing frames
expected_frames = 1:N_json;
missing_frames = setdiff(expected_frames, present_frames);
is_missing = ismember(expected_frames, missing_frames);

%% (B3) Create artificial time vectors for OptiTrack vs JSON
time_opt  = linspace(0, T, N_opt);   % e.g., 218 samples
time_json = linspace(0, T, N_json); % e.g., 768 samples

%% (C) Transform OptiTrack Data to OpenCV Frame
R_alignment = [1,  0,  0; 
               0, -1,  0; 
               0,  0, -1];

R_tgt_in_cam_opt_cell  = cell(N_opt,1);
t_tgt_in_cam_opt_array = zeros(N_opt,3);
R_cam_in_pnp_opt_cell  = cell(N_opt,1);

cam_pos_opencv = zeros(N_opt,3);
tgt_pos_opencv = zeros(N_opt,3);

for i = 1:N_opt
    R_cam_in_pnp_opt_cell{i} = R_alignment * R_cam_in_world(:,:,i) * R_alignment';
    R_tgt_in_world_aligned = R_alignment * R_tgt_in_world(:,:,i) * R_alignment';
    R_tgt_in_cam_opt_cell{i} = R_cam_in_pnp_opt_cell{i}' * R_tgt_in_world_aligned;
    cam_pos_opencv(i,:) = (R_alignment * cam_pos_world(i,:).')';
    tgt_pos_opencv(i,:) = (R_alignment * tgt_pos_world(i,:).')';
    t_tgt_in_cam_opt_array(i,:) = (R_cam_in_pnp_opt_cell{i}' * (tgt_pos_opencv(i,:) - cam_pos_opencv(i,:)).')';
end

%% (D) Resample the OptiTrack data to match the JSON frames count
t_tgt_in_cam_opt_resampled = zeros(N_json,3);
for dim = 1:3
    t_tgt_in_cam_opt_resampled(:,dim) = ...
        interp1(time_opt, t_tgt_in_cam_opt_array(:,dim), time_json, 'linear');
end

eul_opt_temp = zeros(N_opt,3);
for i = 1:N_opt
    eul_opt_temp(i,:) = rotm2eul(R_tgt_in_cam_opt_cell{i}, 'XYZ');
end

eul_opt_resampled = zeros(N_json,3);
for dim = 1:3
    eul_opt_resampled(:,dim) = ...
        interp1(time_opt, eul_opt_temp(:,dim), time_json, 'linear');
end

R_tgt_in_cam_opt_resampled_cell = cell(N_json,1);
for i = 1:N_json
    R_tgt_in_cam_opt_resampled_cell{i} = eul2rotm(eul_opt_resampled(i,:), 'XYZ');
end

N = N_json;

t_tgt_in_cam_opt = t_tgt_in_cam_opt_resampled;
R_tgt_in_cam_opt = R_tgt_in_cam_opt_resampled_cell;

%% (F) Plotting Comparison
figure; hold on; grid on; axis equal;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('3D Relative Poses with Cameras and Axes (Resampled)');

scale = 0.1;
for i = 1:N
    if is_missing(i)
        % Mark missing frames
        plot3(t_tgt_in_cam_opt(i,1), t_tgt_in_cam_opt(i,2), t_tgt_in_cam_opt(i,3), 's', ...
              'Color', [1, 0, 0], 'MarkerSize', 10, 'LineWidth', 1.5);
    else
        % Regular frames
        plot3(t_tgt_in_cam_opt(i,1), t_tgt_in_cam_opt(i,2), t_tgt_in_cam_opt(i,3), 'o', ...
              'Color', [0, 0.447, 0.741], 'MarkerFaceColor', [0, 0.447, 0.741]);
    end

    quiver3(t_tgt_in_cam_opt(i,1), t_tgt_in_cam_opt(i,2), t_tgt_in_cam_opt(i,3), ...
            scale*R_tgt_in_cam_opt{i}(1,1), scale*R_tgt_in_cam_opt{i}(2,1), scale*R_tgt_in_cam_opt{i}(3,1), 'Color', [1,1,0]);
end

legend({'Missing Frame', 'Present Frame'});
hold off;

%% (G) Compare Pose Estimation and OptiTrack Results (same length now)
eul_opt = zeros(N,3);
eul_alg = zeros(N,3);

for i = 1:N
    if ~is_missing(i)  % Only compute Euler angles for present frames
        eul_opt(i,:) = rotm2eul(R_tgt_in_cam_opt{i}, 'XYZ');
        eul_alg(i,:) = rotm2eul(R_tgt_in_cam_alg{i}, 'XYZ');
    else
        eul_opt(i,:) = NaN;  % Mark missing frames as NaN
        eul_alg(i,:) = NaN;
    end
end

figure;

% Subplot 1: Position Comparison
subplot(2,1,1); hold on; grid on;

plot(find(~is_missing), t_tgt_in_cam_opt(~is_missing,1), 'b--', 'LineWidth', 1.5);
plot(find(~is_missing), t_tgt_in_cam_alg(~is_missing,1), 'b-', 'LineWidth', 1.5);
plot(find(~is_missing), t_tgt_in_cam_opt(~is_missing,2), 'g--', 'LineWidth', 1.5);
plot(find(~is_missing), t_tgt_in_cam_alg(~is_missing,2), 'g-', 'LineWidth', 1.5);
plot(find(~is_missing), t_tgt_in_cam_opt(~is_missing,3), 'r--', 'LineWidth', 1.5);
plot(find(~is_missing), t_tgt_in_cam_alg(~is_missing,3), 'r-', 'LineWidth', 1.5);

for i = find(is_missing)
    plot(i, t_tgt_in_cam_alg(i,1), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    plot(i, t_tgt_in_cam_alg(i,2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    plot(i, t_tgt_in_cam_alg(i,3), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
end

xlabel('Frame');
ylabel('Relative Position (m)');
title('Position Comparison (Resampled OptiTrack vs. Pose Estimation)');
legend({'OptiTrack X','Est X','OptiTrack Y','Est Y','OptiTrack Z','Est Z', 'Missing Frames'}, ...
       'Location','best');

% Subplot 2: Orientation Comparison
subplot(2,1,2); hold on; grid on;

plot(find(~is_missing), rad2deg(eul_opt(~is_missing,1)), 'b--', 'LineWidth',1.5);
plot(find(~is_missing), rad2deg(eul_alg(~is_missing,1)), 'b-', 'LineWidth',1.5);
plot(find(~is_missing), rad2deg(eul_opt(~is_missing,2)), 'g--', 'LineWidth',1.5);
plot(find(~is_missing), rad2deg(eul_alg(~is_missing,2)), 'g-', 'LineWidth',1.5);
plot(find(~is_missing), rad2deg(eul_opt(~is_missing,3)), 'm--', 'LineWidth',1.5);
plot(find(~is_missing), rad2deg(eul_alg(~is_missing,3)), 'm-', 'LineWidth',1.5);

for i = find(is_missing)
    plot(i, rad2deg(eul_alg(i,1)), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    plot(i, rad2deg(eul_alg(i,2)), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    plot(i, rad2deg(eul_alg(i,3)), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
end

xlabel('Frame');
ylabel('Angle (deg)');
title('Orientation Comparison (Resampled OptiTrack vs. Pose Estimation)');
legend({'OptiTrack Roll','Est Roll','OptiTrack Pitch','Est Pitch','OptiTrack Yaw','Est Yaw', 'Missing Frames'}, ...
       'Location','best');