function plot_relative_poses_with_cameras_and_axes_resampled_regression()
    %% (0) Preparations
    clear all;
    close all;
    clc;

    % Example total duration in seconds (if known or assumed).
    % Adjust T to match roughly the total time spanned by your data.
    T = 22;  

    % File paths (adjust as needed)
    jsonPath   = '20250107_test2_thresh_15cm.json';   % JSON file with pose estimates + regression fields
    folderPath = '20250107_test2.db3';               % ROS2 bag file with ground truth

    %% (A) Load ground truth (OptiTrack) from ROS2 bag
    bagReader = ros2bagreader(folderPath);
    msgs      = readMessages(bagReader);
    N_opt     = length(msgs);  % total frames in bag

    cam_pos_world  = zeros(N_opt,3);
    cam_quat_world = zeros(N_opt,4);
    tgt_pos_world  = zeros(N_opt,3);
    tgt_quat_world = zeros(N_opt,4);

    for i = 1:N_opt
        d = msgs{i}.data;  % Adjust indexing if needed
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

    %% (B) Load the algorithm’s estimated data + extra fields for regression
    jsonText = fileread(jsonPath);
    algData  = jsondecode(jsonText);

    N_json = length(algData);  % frames in the JSON

    % Preallocate arrays/cells for pose
    R_tgt_in_cam_alg = cell(N_json,1);
    t_tgt_in_cam_alg = zeros(N_json,3);

    % Preallocate for regression-related fields
    num_inliers              = zeros(N_json,1);
    total_matches            = zeros(N_json,1);
    inlier_ratio             = zeros(N_json,1);
    mean_reprojection_error  = zeros(N_json,1);
    std_reprojection_error   = zeros(N_json,1);
    mconf                    = zeros(N_json,1);

    % (Optional) If you want to analyze matched points from JSON
    matched_points_cell      = cell(N_json,1);  % or whatever structure is needed

    for i = 1:N_json
        % Rotation/Translation (adjust field names as needed)
        %  e.g.: algData(i).object_rotation_in_cam, object_translation_in_cam
        R_arr = algData(i).object_rotation_in_cam;
        t_arr = algData(i).object_translation_in_cam;

        R_tgt_in_cam_alg{i} = reshape(R_arr, [3,3]);
        t_tgt_in_cam_alg(i,:) = t_arr;

        % Independent variables from JSON
        num_inliers(i)             = algData(i).num_inliers;
        total_matches(i)           = algData(i).total_matches;
        inlier_ratio(i)            = algData(i).inlier_ratio;
        mean_reprojection_error(i) = algData(i).mean_reprojection_error;
        std_reprojection_error(i)  = algData(i).std_reprojection_error;

        % If 'mconf' is an array per frame, take the average
        mconfVals   = algData(i).mconf;  
        mconf(i)    = mean(mconfVals);

        % Example: store matched points for further analysis
        % (Assume algData(i).matched_points is Nx2 or Nx4, etc.)
        if isfield(algData(i), 'matched_points')
            matched_points_cell{i} = algData(i).matched_points; 
        end
    end

    %% (B2) Create artificial time vectors for OptiTrack vs JSON
    time_opt  = linspace(0, T, N_opt);  
    time_json = linspace(0, T, N_json);

    %% (C) Transform OptiTrack Data to OpenCV Frame
    R_alignment = [ 1,  0,  0; 
                    0, -1,  0; 
                    0,  0, -1 ];

    R_tgt_in_cam_opt_cell  = cell(N_opt,1);
    t_tgt_in_cam_opt_array = zeros(N_opt,3);
    R_cam_in_pnp_opt_cell  = cell(N_opt,1);

    cam_pos_opencv = zeros(N_opt,3);
    tgt_pos_opencv = zeros(N_opt,3);

    for i = 1:N_opt
        R_cam_in_pnp_opt_cell{i} = R_alignment * R_cam_in_world(:,:,i) * R_alignment';
        R_tgt_in_world_aligned   = R_alignment * R_tgt_in_world(:,:,i) * R_alignment';

        R_tgt_in_cam_opt_cell{i} = R_cam_in_pnp_opt_cell{i}' * R_tgt_in_world_aligned;

        cam_pos_opencv(i,:) = (R_alignment * cam_pos_world(i,:).')';
        tgt_pos_opencv(i,:) = (R_alignment * tgt_pos_world(i,:).')';

        t_tgt_in_cam_opt_array(i,:) = ( ...
            R_cam_in_pnp_opt_cell{i}' * (tgt_pos_opencv(i,:) - cam_pos_opencv(i,:)).' ...
        )';
    end

    %% (D) Resample the OptiTrack data to match the JSON frame count
    t_tgt_in_cam_opt_resampled = zeros(N_json,3);
    for dim = 1:3
        t_tgt_in_cam_opt_resampled(:,dim) = ...
            interp1(time_opt, t_tgt_in_cam_opt_array(:,dim), time_json, 'linear');
    end

    % Interpolate rotation in Euler space
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

    %% (E) Overwrite or rename for convenience (N = N_json)
    N = N_json;
    t_tgt_in_cam_opt = t_tgt_in_cam_opt_resampled;        
    R_tgt_in_cam_opt = R_tgt_in_cam_opt_resampled_cell;    

    %% (F) Plot 3D relative poses (Optional)
    figure; hold on; grid on; axis equal;
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('3D Relative Poses (Resampled OptiTrack vs Pose Estimation)');

    scale = 0.1;
    for i = 1:N
        % Resampled OptiTrack target
        plot3(t_tgt_in_cam_opt(i,1), ...
              t_tgt_in_cam_opt(i,2), ...
              t_tgt_in_cam_opt(i,3), ...
              'o', 'Color', [0, 0.447, 0.741], ...
              'MarkerFaceColor', [0, 0.447, 0.741]);

        quiver3(t_tgt_in_cam_opt(i,1), t_tgt_in_cam_opt(i,2), t_tgt_in_cam_opt(i,3), ...
                scale*R_tgt_in_cam_opt{i}(1,1), ...
                scale*R_tgt_in_cam_opt{i}(2,1), ...
                scale*R_tgt_in_cam_opt{i}(3,1), 'Color', [1,1,0]);

        % Pose Estimation target
        plot3(t_tgt_in_cam_alg(i,1), ...
              t_tgt_in_cam_alg(i,2), ...
              t_tgt_in_cam_alg(i,3), ...
              'x', 'Color', [0.85, 0.325, 0.098], ...
              'MarkerSize', 8, 'LineWidth', 1.5);

        quiver3(t_tgt_in_cam_alg(i,1), t_tgt_in_cam_alg(i,2), t_tgt_in_cam_alg(i,3), ...
                scale*R_tgt_in_cam_alg{i}(1,1), ...
                scale*R_tgt_in_cam_alg{i}(2,1), ...
                scale*R_tgt_in_cam_alg{i}(3,1), 'Color', [1,0,0]);
    end

    legend({'Resampled OptiTrack Target', 'Pose Estimation Target'}, 'Location','best');
    hold off;

    %% (G) Compare Pose Estimation vs OptiTrack
    eul_opt = zeros(N,3);
    eul_alg = zeros(N,3);

    for i = 1:N
        eul_opt(i,:) = rotm2eul(R_tgt_in_cam_opt{i}, 'XYZ');
        eul_alg(i,:) = rotm2eul(R_tgt_in_cam_alg{i}, 'XYZ');
    end

    figure;
    subplot(2,1,1); hold on; grid on;
    plot(1:N, t_tgt_in_cam_opt(:,1), 'b--','LineWidth',1.5);
    plot(1:N, t_tgt_in_cam_alg(:,1), 'b-','LineWidth',1.5);
    plot(1:N, t_tgt_in_cam_opt(:,2), 'g--','LineWidth',1.5);
    plot(1:N, t_tgt_in_cam_alg(:,2), 'g-','LineWidth',1.5);
    plot(1:N, t_tgt_in_cam_opt(:,3), 'r--','LineWidth',1.5);
    plot(1:N, t_tgt_in_cam_alg(:,3), 'r-','LineWidth',1.5);
    xlabel('Frame'); ylabel('Position (m)');
    title('Position Comparison');
    legend({'OptiTrack X','Est X','OptiTrack Y','Est Y','OptiTrack Z','Est Z'}, 'Location','best');

    subplot(2,1,2); hold on; grid on;
    plot(1:N, rad2deg(eul_opt(:,1)), 'b--','LineWidth',1.5);
    plot(1:N, rad2deg(eul_alg(:,1)), 'b-','LineWidth',1.5);
    plot(1:N, rad2deg(eul_opt(:,2)), 'g--','LineWidth',1.5);
    plot(1:N, rad2deg(eul_alg(:,2)), 'g-','LineWidth',1.5);
    plot(1:N, rad2deg(eul_opt(:,3)), 'm--','LineWidth',1.5);
    plot(1:N, rad2deg(eul_alg(:,3)), 'm-','LineWidth',1.5);
    xlabel('Frame'); ylabel('Angle (deg)');
    title('Orientation Comparison');
    legend({'OptiTrack Roll','Est Roll','OptiTrack Pitch','Est Pitch','OptiTrack Yaw','Est Yaw'}, 'Location','best');

    %% (H) Compute Errors
    pos_error = t_tgt_in_cam_opt - t_tgt_in_cam_alg;  % [N x 3]
    eul_error = eul_opt - eul_alg;                    % [N x 3] in radians

    figure;
    subplot(3,1,1);
    plot(1:N, pos_error(:,1), 'r-','LineWidth',1.5); grid on;
    xlabel('Frame'); ylabel('Error X (m)');
    title('Position Error in X');

    subplot(3,1,2);
    plot(1:N, pos_error(:,2), 'g-','LineWidth',1.5); grid on;
    xlabel('Frame'); ylabel('Error Y (m)');
    title('Position Error in Y');

    subplot(3,1,3);
    plot(1:N, pos_error(:,3), 'b-','LineWidth',1.5); grid on;
    xlabel('Frame'); ylabel('Error Z (m)');
    title('Position Error in Z');

    figure;
    subplot(3,1,1);
    plot(1:N, rad2deg(eul_error(:,1)), 'r-','LineWidth',1.5); grid on;
    xlabel('Frame'); ylabel('Error Roll (deg)');
    title('Orientation Error in Roll');

    subplot(3,1,2);
    plot(1:N, rad2deg(eul_error(:,2)), 'g-','LineWidth',1.5); grid on;
    xlabel('Frame'); ylabel('Error Pitch (deg)');
    title('Orientation Error in Pitch');

    subplot(3,1,3);
    plot(1:N, rad2deg(eul_error(:,3)), 'b-','LineWidth',1.5); grid on;
    xlabel('Frame'); ylabel('Error Yaw (deg)');
    title('Orientation Error in Yaw');

    %% (I) REGRESSION ANALYSIS (Simple Linear Model)
    X = [num_inliers, total_matches, inlier_ratio, ...
         mean_reprojection_error, std_reprojection_error, mconf];

    % Prepare each dependent variable (Y) - position errors
    Yx = pos_error(:,1); 
    Yy = pos_error(:,2);
    Yz = pos_error(:,3);

    % Orientation errors in degrees
    rollErrorDeg  = rad2deg(eul_error(:,1));
    pitchErrorDeg = rad2deg(eul_error(:,2));
    yawErrorDeg   = rad2deg(eul_error(:,3));

    % ----- Perform Regressions -----
    disp('=== Regression for X-Position Error ===');
    model_x = fitlm(X, Yx, ...
        'VarNames', {'num\_inliers','total\_matches','inlier\_ratio', ...
                     'mean\_reprj','std\_reprj','mconf','posErrorX'});
    disp(model_x);

    disp('=== Regression for Y-Position Error ===');
    model_y = fitlm(X, Yy, ...
        'VarNames', {'num\_inliers','total\_matches','inlier\_ratio', ...
                     'mean\_reprj','std\_reprj','mconf','posErrorY'});
    disp(model_y);

    disp('=== Regression for Z-Position Error ===');
    model_z = fitlm(X, Yz, ...
        'VarNames', {'num\_inliers','total\_matches','inlier\_ratio', ...
                     'mean\_reprj','std\_reprj','mconf','posErrorZ'});
    disp(model_z);

    disp('=== Regression for Roll Error (deg) ===');
    model_roll = fitlm(X, rollErrorDeg, ...
        'VarNames', {'num\_inliers','total\_matches','inlier\_ratio', ...
                     'mean\_reprj','std\_reprj','mconf','rollErrorDeg'});
    disp(model_roll);

    disp('=== Regression for Pitch Error (deg) ===');
    model_pitch = fitlm(X, pitchErrorDeg, ...
        'VarNames', {'num\_inliers','total\_matches','inlier\_ratio', ...
                     'mean\_reprj','std\_reprj','mconf','pitchErrorDeg'});
    disp(model_pitch);

    disp('=== Regression for Yaw Error (deg) ===');
    model_yaw = fitlm(X, yawErrorDeg, ...
        'VarNames', {'num\_inliers','total\_matches','inlier\_ratio', ...
                     'mean\_reprj','std\_reprj','mconf','yawErrorDeg'});
    disp(model_yaw);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% (J) CORRELATION ANALYSIS & SCATTER PLOTS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Combine all variables for correlation check:
    % X: [num_inliers, total_matches, ..., mconf]
    % Y: [Yx, Yy, Yz, rollErrorDeg, pitchErrorDeg, yawErrorDeg]
    allVars = [X, Yx, Yy, Yz, rollErrorDeg, pitchErrorDeg, yawErrorDeg];
    % You might want to label them carefully:
    varLabels = {'inliers','matches','inlierRatio','meanReprj','stdReprj','mconf', ...
                 'Xerr','Yerr','Zerr','RollErr','PitchErr','YawErr'};

    % 1) Correlation Matrix
    corrMat = corrcoef(allVars);  % Pearson correlation by default
    figure;
    imagesc(corrMat);
    colorbar;
    set(gca, 'XTick', 1:length(varLabels), 'XTickLabel', varLabels, ...
             'YTick', 1:length(varLabels), 'YTickLabel', varLabels);
    title('Correlation Matrix (Pearson)');

    % 2) Scatter Plot Matrix
    figure;
    plotmatrix(allVars);
    sgtitle('Scatter Plot Matrix of All Variables');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% (K) STEPWISE REGRESSION (Example for X-Error)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stepwise regression tries adding/removing terms automatically
    % 'linear' or 'interactions', 'quadratic' etc. can be tested
    disp('=== Stepwise Regression for X-Position Error ===');
    stepModelX = stepwiselm(X, Yx, 'linear', ...
        'VarNames', {'inliers','matches','ratio','meanReprj','stdReprj','mconf','Xerr'});
    disp(stepModelX);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% (L) TRY INTERACTION TERMS (Example for Y-Position Error)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % For instance, we can include all pairwise interactions:
    disp('=== Interaction Model for Y-Position Error ===');
    model_y_inter = fitlm(X, Yy, 'interactions', ...
        'VarNames', {'inliers','matches','ratio','meanReprj','stdReprj','mconf','Yerr'});
    disp(model_y_inter);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% (M) EXAMPLE: RANDOM FOREST REGRESSION (for Z-Error)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Using MATLAB's TreeBagger (ensemble of regression trees)
    % NOTE: This is just a simple example; consider hyperparameter tuning.
    disp('=== Random Forest (TreeBagger) for Z-Position Error ===');
    nTrees = 50;  % or more
    rfModelZ = TreeBagger(nTrees, X, Yz, 'Method','regression');
    % We can predict:
    zPred = predict(rfModelZ, X);
    % Compute MSE or R^2 manually:
    mseZ = mean((zPred - Yz).^2);
    SSres = sum((zPred - Yz).^2);
    SStot = sum((Yz - mean(Yz)).^2);
    r2Z = 1 - SSres/SStot;

    disp(['  RandomForest MSE(Z) = ', num2str(mseZ)]);
    disp(['  RandomForest R2(Z)  = ', num2str(r2Z)]);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% (N) ANALYZE MATCHED POINTS (if available in JSON)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Suppose algData(i).matched_points = Nx2 (or Nx4, etc.).
    % We can do a basic summary, e.g. average number of matched points:
    hasMatches = cellfun(@(c) ~isempty(c), matched_points_cell);
    matchCounts = zeros(N,1);
    for i = 1:N
        if hasMatches(i)
            matchCounts(i) = size(matched_points_cell{i},1);  % number of matched points
        else
            matchCounts(i) = 0;
        end
    end

    figure;
    plot(1:N, matchCounts, 'ko-','LineWidth',1.5);
    xlabel('Frame'); ylabel('Num Matched Points');
    title('Number of Matched Points per Frame');

    % If you want to see distribution of matched points across all frames:
    figure;
    histogram(matchCounts, 'BinMethod','integers');
    xlabel('Number of Matched Points'); ylabel('Frequency');
    title('Histogram of Matched Points Count');

    % You could also analyze the geometry of matched_points_cell, etc.
    % For example, if matched_points_cell{i} = Nx2 pixel coords,
    % you can compute their bounding box or average location per frame, etc.

end