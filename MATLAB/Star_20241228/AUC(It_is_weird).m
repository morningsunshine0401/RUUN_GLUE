function plot_relative_poses_with_cameras_and_axes_GPT()
    %% (0) Preparations
    clear all;
    close all;
    clc;
    jsonPath   = '20241227_C_test4.json';   % JSON file (estimated poses)
    folderPath = '20241227_test4.db3';      % ROS2 bag (.db3) file (GT data)

    %% (A) Load ground truth from ROS2 bag
    bagReader  = ros2bagreader(folderPath);
    msgs       = readMessages(bagReader);
    N = length(msgs);  % Total frames in bag

    cam_pos_world  = zeros(N,3);
    cam_quat_world = zeros(N,4);
    tgt_pos_world  = zeros(N,3);
    tgt_quat_world = zeros(N,4);

    for i = 1:N
        d = msgs{i}.data;
        cam_pos_world(i,:)  = d(1:3);
        cam_quat_world(i,:) = d(4:7);
        tgt_pos_world(i,:)  = d(8:10);
        tgt_quat_world(i,:) = d(11:14);
    end

    %% Convert quaternions to rotation matrices
    R_cam_in_world = zeros(3,3,N);
    R_tgt_in_world = zeros(3,3,N);
    for i = 1:N
        R_cam_in_world(:,:,i) = quat2rotm(cam_quat_world(i,:));
        R_tgt_in_world(:,:,i) = quat2rotm(tgt_quat_world(i,:));
    end

    %% (B) Load the algorithm’s estimated data from JSON
    jsonText = fileread(jsonPath);
    algData  = jsondecode(jsonText);

    R_tgt_in_cam_alg = cell(N,1);
    t_tgt_in_cam_alg = zeros(N,3);

    for i = 1:N
        R_arr = algData(i).kf_rotation_matrix;
        t_arr = algData(i).kf_translation_vector;

        R_tgt_in_cam_alg{i} = reshape(R_arr, [3,3]);
        t_tgt_in_cam_alg(i,:) = t_arr;
    end

    %% (C) Transform OptiTrack Data to OpenCV Frame
    R_alignment = [1,  0,  0; 
                  0, -1,  0; 
                  0,  0, -1];
               
    R_tgt_in_cam_opt = cell(N,1);
    t_tgt_in_cam_opt = zeros(N,3);
    R_cam_in_pnp_opt = cell(N,1);
    
    % Pre-allocate aligned positions
    cam_pos_opencv = zeros(N,3);
    tgt_pos_opencv = zeros(N,3);

    for i = 1:N
        %% 1) Align Camera Rotation
        R_cam_in_pnp_opt{i} = R_alignment * R_cam_in_world(:,:,i) * R_alignment';

        %% 2) Align Target Rotation
        R_tgt_in_world_aligned = R_alignment * R_tgt_in_world(:,:,i) * R_alignment';

        %% 3) Compute Target Rotation in Camera Frame
        R_tgt_in_cam_opt{i} = R_cam_in_pnp_opt{i}' * R_tgt_in_world_aligned;

        %% 4) Align Positions to OpenCV Frame
        cam_pos_opencv(i,:) = (R_alignment * cam_pos_world(i,:)')';
        tgt_pos_opencv(i,:) = (R_alignment * tgt_pos_world(i,:)')';

        %% 5) Compute Relative Position in Camera Frame
        t_tgt_in_cam_opt(i,:) = ( ...
            R_cam_in_pnp_opt{i}' * (tgt_pos_opencv(i,:) - cam_pos_opencv(i,:))' ...
        )';
    end

    %% (D) Plot relative poses in 3D
    figure; hold on; grid on; axis equal;
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    title('3D Relative Poses with Cameras and Axes');
    
    % Scale for the axes/quivers
    scale = 0.1;
    
    for i = 1:N
        % Plot OptiTrack Camera Pose and Axes
        quiver3(0, 0, 0, scale*R_cam_in_pnp_opt{i}(1,1), scale*R_cam_in_pnp_opt{i}(2,1), scale*R_cam_in_pnp_opt{i}(3,1), 'Color', [1, 1, 0]); % X-axis: Yellow
        quiver3(0, 0, 0, scale*R_cam_in_pnp_opt{i}(1,2), scale*R_cam_in_pnp_opt{i}(2,2), scale*R_cam_in_pnp_opt{i}(3,2), 'Color', [1, 0.5, 0]); % Y-axis: Orange
        quiver3(0, 0, 0, scale*R_cam_in_pnp_opt{i}(1,3), scale*R_cam_in_pnp_opt{i}(2,3), scale*R_cam_in_pnp_opt{i}(3,3), 'Color', [0, 0, 0]); % Z-axis: Black

        % Plot OptiTrack Target Pose and Axes
        plot3(t_tgt_in_cam_opt(i,1), t_tgt_in_cam_opt(i,2), t_tgt_in_cam_opt(i,3), 'o', 'Color', [0, 0.447, 0.741], 'MarkerFaceColor', [0, 0.447, 0.741]); % Target Position
        quiver3(t_tgt_in_cam_opt(i,1), t_tgt_in_cam_opt(i,2), t_tgt_in_cam_opt(i,3), scale*R_tgt_in_cam_opt{i}(1,1), scale*R_tgt_in_cam_opt{i}(2,1), scale*R_tgt_in_cam_opt{i}(3,1), 'Color', [1, 1, 0]); % X-axis: Yellow
        quiver3(t_tgt_in_cam_opt(i,1), t_tgt_in_cam_opt(i,2), t_tgt_in_cam_opt(i,3), scale*R_tgt_in_cam_opt{i}(1,2), scale*R_tgt_in_cam_opt{i}(2,2), scale*R_tgt_in_cam_opt{i}(3,2), 'Color', [1, 0.5, 0]); % Y-axis: Orange
        quiver3(t_tgt_in_cam_opt(i,1), t_tgt_in_cam_opt(i,2), t_tgt_in_cam_opt(i,3), scale*R_tgt_in_cam_opt{i}(1,3), scale*R_tgt_in_cam_opt{i}(2,3), scale*R_tgt_in_cam_opt{i}(3,3), 'Color', [0, 0, 0]); % Z-axis: Black

        % Plot Pose Estimation Camera Pose and Axes
        plot3(0, 0, 0, 'x', 'Color', [1, 0, 0], 'MarkerSize', 8, 'LineWidth', 1.5); % Camera Origin
        quiver3(0, 0, 0, scale, 0, 0, 'Color', [1, 0, 0]); % X-axis: Red
        quiver3(0, 0, 0, 0, scale, 0, 'Color', [0, 1, 0]); % Y-axis: Green
        quiver3(0, 0, 0, 0, 0, scale, 'Color', [0, 0, 1]); % Z-axis: Blue

        % Plot Pose Estimation Target Pose and Axes
        plot3(t_tgt_in_cam_alg(i,1), t_tgt_in_cam_alg(i,2), t_tgt_in_cam_alg(i,3), 'x', 'Color', [0.85, 0.325, 0.098], 'MarkerSize', 8, 'LineWidth', 1.5); % Target Position
        quiver3(t_tgt_in_cam_alg(i,1), t_tgt_in_cam_alg(i,2), t_tgt_in_cam_alg(i,3), scale*R_tgt_in_cam_alg{i}(1,1), scale*R_tgt_in_cam_alg{i}(2,1), scale*R_tgt_in_cam_alg{i}(3,1), 'Color', [1, 0, 0]); % X-axis: Red
        quiver3(t_tgt_in_cam_alg(i,1), t_tgt_in_cam_alg(i,2), t_tgt_in_cam_alg(i,3), scale*R_tgt_in_cam_alg{i}(1,2), scale*R_tgt_in_cam_alg{i}(2,2), scale*R_tgt_in_cam_alg{i}(3,2), 'Color', [0, 1, 0]); % Y-axis: Green
        quiver3(t_tgt_in_cam_alg(i,1), t_tgt_in_cam_alg(i,2), t_tgt_in_cam_alg(i,3), ...
        scale*R_tgt_in_cam_alg{i}(1,3), scale*R_tgt_in_cam_alg{i}(2,3), scale*R_tgt_in_cam_alg{i}(3,3), 'Color', [0, 0, 1]); % Z-axis: Blue
        %quiver3(t_tgt_in_cam_alg(i,1), t_tgt_in_cam_alg(i,2), t_tgt_in_cam_alg{i,3}, scale*R_tgt_in_cam_alg{i}(1,3), scale*R_tgt_in_cam_alg{i}(2,3), scale*R_tgt_in_cam_alg{i}(3,3), 'Color', [0, 0, 1]); % Z-axis: Blue
    end

    legend({'OptiTrack Target', 'Pose Estimation Target', 'OptiTrack Axes', 'Pose Estimation Axes'});
    hold off;

    %% (E) Compare Pose Estimation and OptiTrack Results
    % Compute Orientation Errors Using Geodesic Distance
    orientation_error_deg = zeros(N,1);
    for i = 1:N
        % Ground Truth Rotation (Target in Camera Frame)
        R_gt = R_tgt_in_cam_opt{i};
        % Estimated Rotation (Algorithm)
        R_est = R_tgt_in_cam_alg{i};
        
        % Convert rotation matrices to quaternions
        q_gt = rotm2quat(R_gt);
        q_est = rotm2quat(R_est);
        
        % Compute geodesic distance
        orientation_error_deg(i) = quat_geodesic_distance(q_gt, q_est);
    end

    % Compute Position Errors
    position_error_mag = sqrt(sum((t_tgt_in_cam_opt - t_tgt_in_cam_alg).^2, 2));

    %% (D) Plot Relative Pose Comparisons (Optional: Retain your existing plots)
    %% (E) Compare Pose Estimation and OptiTrack Results
    % Convert rotation matrices to Euler angles (roll, pitch, yaw)
    eul_opt = zeros(N,3); % OptiTrack Euler angles
    eul_alg = zeros(N,3); % Algorithm Euler angles

    for i = 1:N
        eul_opt(i,:) = rotm2eul(R_tgt_in_cam_opt{i}, 'XYZ'); % [roll, pitch, yaw]
        eul_alg(i,:) = rotm2eul(R_tgt_in_cam_alg{i}, 'XYZ'); % [roll, pitch, yaw]
    end

    % Plot Position Comparison (Relative Pose)
    figure;
    subplot(2,1,1);
    hold on; grid on;
    plot(1:N, t_tgt_in_cam_opt(:,1), 'b--', 'LineWidth', 1.5); % OptiTrack Relative X
    plot(1:N, t_tgt_in_cam_alg(:,1), 'b-', 'LineWidth', 1.5); % Estimation Relative X
    plot(1:N, t_tgt_in_cam_opt(:,2), 'g--', 'LineWidth', 1.5); % OptiTrack Relative Y
    plot(1:N, t_tgt_in_cam_alg(:,2), 'g-', 'LineWidth', 1.5); % Estimation Relative Y
    plot(1:N, t_tgt_in_cam_opt(:,3), 'r--', 'LineWidth', 1.5); % OptiTrack Relative Z
    plot(1:N, t_tgt_in_cam_alg(:,3), 'r-', 'LineWidth', 1.5); % Estimation Relative Z
    xlabel('Frame');
    ylabel('Relative Position (m)');
    title('Position Comparison: OptiTrack vs Pose Estimation');
    legend({'OptiTrack Relative X', 'Estimation Relative X', ...
           'OptiTrack Relative Y', 'Estimation Relative Y', ...
           'OptiTrack Relative Z', 'Estimation Relative Z'}, ...
           'Location', 'best');
    hold off;

    % Plot Orientation Comparison
    subplot(2,1,2);
    plot(1:N, rad2deg(eul_opt(:,1)), 'b--', 'LineWidth', 1.5); hold on;
    plot(1:N, rad2deg(eul_opt(:,2)), 'g--', 'LineWidth', 1.5);
    plot(1:N, rad2deg(eul_opt(:,3)), 'm--', 'LineWidth', 1.5);
    plot(1:N, rad2deg(eul_alg(:,1)), 'b-', 'LineWidth', 1.5);
    plot(1:N, rad2deg(eul_alg(:,2)), 'g-', 'LineWidth', 1.5);
    plot(1:N, rad2deg(eul_alg(:,3)), 'm-', 'LineWidth', 1.5);
    xlabel('Frame');
    ylabel('Angle (degrees)');
    title('Orientation Comparison: OptiTrack vs Pose Estimation');
    legend({'OptiTrack Roll', 'OptiTrack Pitch', 'OptiTrack Yaw', ...
           'Estimation Roll', 'Estimation Pitch', 'Estimation Yaw'}, ...
           'Location', 'best');
    grid on;
    hold off;

    % Plot Errors in Position
    pos_error = t_tgt_in_cam_opt - t_tgt_in_cam_alg;
    figure;
    subplot(3,1,1);
    plot(1:N, pos_error(:,1), 'r-', 'LineWidth', 1.5);
    xlabel('Frame');
    ylabel('Error X (m)');
    title('Position Error in X');
    grid on;

    subplot(3,1,2);
    plot(1:N, pos_error(:,2), 'g-', 'LineWidth', 1.5);
    xlabel('Frame');
    ylabel('Error Y (m)');
    title('Position Error in Y');
    grid on;

    subplot(3,1,3);
    plot(1:N, pos_error(:,3), 'b-', 'LineWidth', 1.5);
    xlabel('Frame');
    ylabel('Error Z (m)');
    title('Position Error in Z');
    grid on;

    % Plot Errors in Orientation
    eul_error = eul_opt - eul_alg;
    figure;
    subplot(3,1,1);
    plot(1:N, rad2deg(eul_error(:,1)), 'r-', 'LineWidth', 1.5);
    xlabel('Frame');
    ylabel('Error Roll (deg)');
    title('Orientation Error in Roll');
    grid on;

    subplot(3,1,2);
    plot(1:N, rad2deg(eul_error(:,2)), 'g-', 'LineWidth', 1.5);
    xlabel('Frame');
    ylabel('Error Pitch (deg)');
    title('Orientation Error in Pitch');
    grid on;

    subplot(3,1,3);
    plot(1:N, rad2deg(eul_error(:,3)), 'b-', 'LineWidth', 1.5);
    xlabel('Frame');
    ylabel('Error Yaw (deg)');
    title('Orientation Error in Yaw');
    grid on;

    %% (F) Compute AUC for Relative Pose Estimation Errors
    % Define thresholds
    orientation_thresholds_deg = [10, 15, 25]; % degrees
    position_thresholds_m = [0.10, 0.15, 0.25]; % meters (adjust as needed)

    % Compute Cumulative Distribution
    [cdf_orientation, sorted_orientation] = ecdf(orientation_error_deg);
    [cdf_position, sorted_position] = ecdf(position_error_mag);

    % Plot Cumulative Distribution for Orientation Errors
    figure;
    subplot(2,1,1);
    plot(sorted_orientation, cdf_orientation, 'b-', 'LineWidth', 2);
    hold on;
    for th = orientation_thresholds_deg
        xline(th, '--r', ['Threshold ', num2str(th), '°'], 'LineWidth', 1.5);
    end
    xlabel('Orientation Error Magnitude (degrees)');
    ylabel('Cumulative Probability');
    title('Cumulative Distribution of Orientation Errors');
    legend(['CDF', arrayfun(@(x) [' Threshold ' num2str(x) '°'], orientation_thresholds_deg, 'UniformOutput', false)], 'Location', 'best');
    grid on;
    hold off;

    % Plot Cumulative Distribution for Position Errors
    subplot(2,1,2);
    plot(sorted_position, cdf_position, 'g-', 'LineWidth', 2);
    hold on;
    for th = position_thresholds_m
        xline(th, '--r', ['Threshold ', num2str(th), 'm'], 'LineWidth', 1.5);
    end
    xlabel('Position Error Magnitude (meters)');
    ylabel('Cumulative Probability');
    title('Cumulative Distribution of Position Errors');
    legend(['CDF', arrayfun(@(x) [' Threshold ' num2str(x) 'm'], position_thresholds_m, 'UniformOutput', false)], 'Location', 'best');
    grid on;
    hold off;

    %% Calculate AUC up to Each Threshold
    % Orientation AUC
    auc_orientation = zeros(length(orientation_thresholds_deg),1);
    for i = 1:length(orientation_thresholds_deg)
        th = orientation_thresholds_deg(i);
        idx = sorted_orientation <= th;
        auc_orientation(i) = trapz(sorted_orientation(idx), cdf_orientation(idx));
        fprintf('AUC for Orientation Errors up to %d°: %.4f\n', th, auc_orientation(i));
    end

    % Position AUC
    auc_position = zeros(length(position_thresholds_m),1);
    for i = 1:length(position_thresholds_m)
        th = position_thresholds_m(i);
        idx = sorted_position <= th;
        auc_position(i) = trapz(sorted_position(idx), cdf_position(idx));
        fprintf('AUC for Position Errors up to %.2fm: %.4f\n', th, auc_position(i));
    end

    %% (Optional) Display AUC Results
    figure;
    subplot(2,1,1);
    bar(orientation_thresholds_deg, auc_orientation);
    xlabel('Orientation Threshold (degrees)');
    ylabel('AUC');
    title('AUC for Orientation Errors');
    grid on;

    subplot(2,1,2);
    bar(position_thresholds_m, auc_position);
    xlabel('Position Threshold (meters)');
    ylabel('AUC');
    title('AUC for Position Errors');
    grid on;

end

% Auxiliary Function: Geodesic Distance Between Two Quaternions
function angle_deg = quat_geodesic_distance(q1, q2)
    % Ensure quaternions are unit quaternions
    q1 = q1 / norm(q1);
    q2 = q2 / norm(q2);
    
    % Compute the dot product
    dot_product = abs(dot(q1, q2));
    
    % Clamp the dot product to avoid numerical issues
    dot_product = min(max(dot_product, -1.0), 1.0);
    
    % Compute the angle in radians
    angle_rad = 2 * acos(dot_product);
    
    % Convert to degrees
    angle_deg = rad2deg(angle_rad);
end
