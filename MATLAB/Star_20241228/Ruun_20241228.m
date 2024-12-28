function plot_relative_poses_with_cameras_and_axes()
    %% (0) Preparations
    clear all;
    close all;
    clc;
    jsonPath   = '20241227_C_test4.json';   % JSON file (estimated poses)
    folderPath = '20241227_test4.db3';      % ROS2 bag (.db3) file (GT data)

    %% (A) Load ground truth from ROS2 bag
    bagReader  = ros2bagreader(folderPath);
    msgs       = readMessages(bagReader);
    N = length(msgs);  % total frames in bag

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
        quiver3(t_tgt_in_cam_alg(i,1), t_tgt_in_cam_alg(i,2), t_tgt_in_cam_alg(i,3), scale*R_tgt_in_cam_alg{i}(1,3), scale*R_tgt_in_cam_alg{i}(2,3), scale*R_tgt_in_cam_alg{i}(3,3), 'Color', [0, 0, 1]); % Z-axis: Blue
    end
    
    legend({'OptiTrack Target', 'Pose Estimation Target', 'OptiTrack Axes', 'Pose Estimation Axes'});
    hold off;

