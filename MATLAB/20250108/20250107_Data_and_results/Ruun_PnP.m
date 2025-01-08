function plot_relative_poses_with_cameras_and_axes_resampled_regression()
    %% (0) Preparations
    clear all;
    close all;
    clc;

    T = 22;  % Adjust duration as per your data

    % File paths (adjust as needed)
    jsonPath   = '20250108_test2_pnp_thresh.json';
    folderPath = '20250107_test2.db3';

    %% (A) Load ground truth (OptiTrack) from ROS2 bag
    bagReader = ros2bagreader(folderPath);
    msgs      = readMessages(bagReader);
    N_opt     = length(msgs);

    % Ground truth data initialization
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

    %% (B) Load algorithm’s estimated data (JSON)
    jsonText = fileread(jsonPath);
    algData  = jsondecode(jsonText);
    N_json   = length(algData);

    % Estimated data initialization
    R_tgt_in_cam_alg = cell(N_json,1);
    t_tgt_in_cam_alg = zeros(N_json,3);

    % Additional fields for analysis
    num_inliers              = zeros(N_json,1);
    total_matches            = zeros(N_json,1);
    inlier_ratio             = zeros(N_json,1);
    mean_reprojection_error  = zeros(N_json,1);
    std_reprojection_error   = zeros(N_json,1);
    mconf                    = zeros(N_json,1);
    region_front_right       = zeros(N_json,1);
    region_front_left        = zeros(N_json,1);
    region_back_right        = zeros(N_json,1);
    region_back_left         = zeros(N_json,1);
    coverage_score           = zeros(N_json,1);

    for i = 1:N_json
        R_arr = algData(i).object_rotation_in_cam;
        t_arr = algData(i).object_translation_in_cam;

        R_tgt_in_cam_alg{i} = reshape(R_arr, [3,3]);
        t_tgt_in_cam_alg(i,:) = t_arr;

        num_inliers(i)             = algData(i).num_inliers;
        total_matches(i)           = algData(i).total_matches;
        inlier_ratio(i)            = algData(i).inlier_ratio;
        mean_reprojection_error(i) = algData(i).mean_reprojection_error;
        std_reprojection_error(i)  = algData(i).std_reprojection_error;

        if isfield(algData(i), 'mconf')
            mconf(i) = mean(algData(i).mconf);
        end

        if isfield(algData(i), 'region_distribution')
            region_front_right(i) = algData(i).region_distribution.front_right;
            region_front_left(i)  = algData(i).region_distribution.front_left;
            region_back_right(i)  = algData(i).region_distribution.back_right;
            region_back_left(i)   = algData(i).region_distribution.back_left;
        end

        if isfield(algData(i), 'coverage_score')
            coverage_score(i) = algData(i).coverage_score;
        end
    end

    %% (C) Transform OptiTrack data to match OpenCV convention
    R_alignment = [1, 0,  0; 
                   0, -1, 0; 
                   0,  0, -1];

    R_tgt_in_cam_opt_cell = cell(N_opt,1);
    t_tgt_in_cam_opt_array = zeros(N_opt,3);

    for i = 1:N_opt
        R_tgt_in_world_aligned = R_alignment * R_tgt_in_world(:,:,i) * R_alignment';
        R_cam_in_pnp_opt = R_alignment * R_cam_in_world(:,:,i) * R_alignment';

        R_tgt_in_cam_opt_cell{i} = R_cam_in_pnp_opt' * R_tgt_in_world_aligned;

        cam_pos_opencv = (R_alignment * cam_pos_world(i,:)')';
        tgt_pos_opencv = (R_alignment * tgt_pos_world(i,:)')';

        t_tgt_in_cam_opt_array(i,:) = ( ...
            R_cam_in_pnp_opt' * (tgt_pos_opencv - cam_pos_opencv)' ...
        )';
    end

    %% (D) Resample ground truth to match JSON data
    time_opt  = linspace(0, T, N_opt);
    time_json = linspace(0, T, N_json);

    t_tgt_in_cam_opt_resampled = interp1(time_opt, t_tgt_in_cam_opt_array, time_json, 'linear');

    eul_opt_temp = zeros(N_opt,3);
    for i = 1:N_opt
        eul_opt_temp(i,:) = rotm2eul(R_tgt_in_cam_opt_cell{i}, 'XYZ');
    end
    eul_opt_resampled = interp1(time_opt, eul_opt_temp, time_json, 'linear');

    %% (E) Compute Errors
    pos_error = t_tgt_in_cam_opt_resampled - t_tgt_in_cam_alg;
    eul_error = eul_opt_resampled - cell2mat(cellfun(@(R) rotm2eul(R, 'XYZ'), R_tgt_in_cam_alg, 'UniformOutput', false));

    %% (F) Regression Analysis
    X = [region_front_right, region_front_left, region_back_right, region_back_left, coverage_score];
    varNames = {'front_right', 'front_left', 'back_right', 'back_left', 'coverage_score'};

    Y = {pos_error(:,1), pos_error(:,2), pos_error(:,3), ...
         rad2deg(eul_error(:,1)), rad2deg(eul_error(:,2)), rad2deg(eul_error(:,3))};
    Y_names = {'Error X', 'Error Y', 'Error Z', 'Error Roll', 'Error Pitch', 'Error Yaw'};

    for i = 1:length(Y)
        disp(['=== Regression for ', Y_names{i}, ' ===']);
        model = fitlm(X, Y{i}, 'VarNames', [varNames, Y_names(i)]);
        disp(model);
    end

    %% (G) Visualize Results
    for i = 1:length(Y)
        figure;
        scatter(coverage_score, Y{i}, 'filled');
        xlabel('Coverage Score');
        ylabel(Y_names{i});
        title(['Coverage Score vs ', Y_names{i}]);
    end
end