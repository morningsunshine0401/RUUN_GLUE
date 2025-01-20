function plot_relative_poses_time_sync()

    %% (1) Preparations
    clear all; close all; clc;

    % Edit these paths to match your data:
    bagFile  = '20250120_test5.db3';   % Bag with /OPTI/rb_infos
    jsonPath = '20250120_test5_1.json';% JSON with your pose estimation

    % Our ground-truth topic in the bag
    topicName = "/OPTI/rb_infos"; 

    %% (2) Read the bag in chronological order
    bagReader = ros2bagreader(bagFile);
    msgList   = bagReader.MessageList;   % Table: {Time, Topic, MessageType}
    allMsgs   = readMessages(bagReader);

    % Find the row indices for our GT topic
    idxOpt = find(msgList.Topic == topicName);
    N_opt  = numel(idxOpt);
    if N_opt < 1
        error("No messages found on topic '%s' in bag '%s'", topicName, bagFile);
    end
    fprintf("Found %d messages on %s\n", N_opt, topicName);

    %% Allocate storage
    cam_pos_world  = zeros(N_opt,3);
    cam_quat_world = zeros(N_opt,4);
    tgt_pos_world  = zeros(N_opt,3);
    tgt_quat_world = zeros(N_opt,4);

    % We'll store numeric timestamps for each GT message
    optTimestamps = zeros(N_opt,1);

    %% (3) Parse GT data + timestamps
    for i = 1:N_opt
        row = idxOpt(i);
        d   = allMsgs{row}.data;  % float[14]: [cam_pos,cam_quat,tgt_pos,tgt_quat]

        cam_pos_world(i,:)   = d(1:3);
        cam_quat_world(i,:)  = d(4:7);
        tgt_pos_world(i,:)   = d(8:10);
        tgt_quat_world(i,:)  = d(11:14);

        % Fill the numeric timestamp
        tVal = msgList.Time(row); 
        if isdatetime(tVal)
            optTimestamps(i) = posixtime(tVal);    % convert datetime -> seconds
        else
            optTimestamps(i) = double(tVal);
        end
    end

    %% (4) Convert quaternions->rotation matrices
    R_cam_in_world = zeros(3,3,N_opt);
    R_tgt_in_world = zeros(3,3,N_opt);

    for i = 1:N_opt
        R_cam_in_world(:,:,i) = quat2rotm(cam_quat_world(i,:));
        R_tgt_in_world(:,:,i) = quat2rotm(tgt_quat_world(i,:));
    end

    %% (5) Load the JSON pose estimation
    jsonText = fileread(jsonPath);
    algData  = jsondecode(jsonText);
    N_json   = numel(algData);

    R_tgt_in_cam_alg = cell(N_json,1);
    t_tgt_in_cam_alg = zeros(N_json,3);
    algTimes         = zeros(N_json,1);

    for i = 1:N_json
        R_arr = algData(i).kf_rotation_matrix;   % 9 floats
        t_arr = algData(i).kf_translation_vector;% 3 floats
        R_tgt_in_cam_alg{i} = reshape(R_arr,[3,3]);
        t_tgt_in_cam_alg(i,:) = t_arr;

        % The algorithm's timestamp for time-based matching
        algTimes(i) = algData(i).timestamp;
    end

    %% (6) Apply your coordinate system "like your code"
    % Example: R_alignment flips X and Y, keeps Z
    R_alignment = [-1, 0,  0; 
                   0, -1,  0; 
                   0,  0,  1];

    t_rel_array = zeros(N_opt,3);
    R_rel_array = zeros(3,3,N_opt);

    for i = 1:N_opt
        % Align camera rotation
        R_cam_aligned = R_alignment * R_cam_in_world(:,:,i) * R_alignment';
        % Align target rotation
        R_tgt_aligned = R_alignment * R_tgt_in_world(:,:,i) * R_alignment';

        % Relative rotation: camera->target
        R_rel_array(:,:,i) = R_cam_aligned' * R_tgt_aligned;

        % Align positions
        cam_pos_aligned = (R_alignment * cam_pos_world(i,:).')';
        tgt_pos_aligned = (R_alignment * tgt_pos_world(i,:).')';

        % Relative position in camera frame
        t_rel_array(i,:) = (R_cam_aligned' * (tgt_pos_aligned - cam_pos_aligned).').';
    end

    %% (7) Time-based matching
    matched_t_opt = zeros(N_json,3);
    matched_R_opt = cell(N_json,1);

    for i = 1:N_json
        thisAlgTime = algTimes(i);
        [~, idxClosest] = min(abs(optTimestamps - thisAlgTime));

        matched_t_opt(i,:)  = t_rel_array(idxClosest,:);
        matched_R_opt{i}    = R_rel_array(:,:,idxClosest);
    end

    %% (8) Plot 3D scatter: GT vs. Algorithm
    figure('Name','3D Relative Poses','NumberTitle','off');
    hold on; grid on; axis equal;
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('3D Relative Poses (Time-Synced)');

    scale = 0.1;
    for i = 1:N_json
        % GT
        plot3(matched_t_opt(i,1), matched_t_opt(i,2), matched_t_opt(i,3), ...
              'o','Color',[0,0.447,0.741],'MarkerFaceColor',[0,0.447,0.741]);
        R_opt = matched_R_opt{i};
        quiver3(matched_t_opt(i,1), matched_t_opt(i,2), matched_t_opt(i,3), ...
                scale*R_opt(1,1),scale*R_opt(2,1),scale*R_opt(3,1), 'Color',[1,1,0]);

        % Algorithm
        plot3(t_tgt_in_cam_alg(i,1), t_tgt_in_cam_alg(i,2), t_tgt_in_cam_alg(i,3), ...
              'x','Color',[0.85,0.325,0.098],'MarkerSize',8,'LineWidth',1.5);
        R_alg = R_tgt_in_cam_alg{i};
        quiver3(t_tgt_in_cam_alg(i,1), t_tgt_in_cam_alg(i,2), t_tgt_in_cam_alg(i,3), ...
                scale*R_alg(1,1),scale*R_alg(2,1),scale*R_alg(3,1), 'Color',[1,0,0]);
    end
    legend({'GT Target','GT Orientation','Estimated Target','Est Orientation'});
    hold off;

    %% (9) Plot X/Y/Z comparison
    figure('Name','Position Comparison','NumberTitle','off');
    hold on; grid on;
    N = N_json;
    plot(1:N, matched_t_opt(:,1),'b--','LineWidth',1.5);
    plot(1:N, t_tgt_in_cam_alg(:,1),'b-','LineWidth',1.5);
    plot(1:N, matched_t_opt(:,2),'g--','LineWidth',1.5);
    plot(1:N, t_tgt_in_cam_alg(:,2),'g-','LineWidth',1.5);
    plot(1:N, matched_t_opt(:,3),'r--','LineWidth',1.5);
    plot(1:N, t_tgt_in_cam_alg(:,3),'r-','LineWidth',1.5);
    xlabel('Frame'); ylabel('Relative Position (m)');
    title('Position Comparison (GT vs. Algorithm)');
    legend({'GT X','Est X','GT Y','Est Y','GT Z','Est Z'},'Location','best');

    %% (10) Orientation comparison
    eul_opt = zeros(N,3);
    eul_alg = zeros(N,3);
    for i = 1:N
        eul_opt(i,:) = rotm2eul(matched_R_opt{i}, 'XYZ');
        eul_alg(i,:) = rotm2eul(R_tgt_in_cam_alg{i}, 'XYZ');
    end

    figure('Name','Orientation Comparison','NumberTitle','off');
    hold on; grid on;
    plot(1:N,rad2deg(eul_opt(:,1)),'b--','LineWidth',1.5);
    plot(1:N,rad2deg(eul_alg(:,1)),'b-','LineWidth',1.5);
    plot(1:N,rad2deg(eul_opt(:,2)),'g--','LineWidth',1.5);
    plot(1:N,rad2deg(eul_alg(:,2)),'g-','LineWidth',1.5);
    plot(1:N,rad2deg(eul_opt(:,3)),'m--','LineWidth',1.5);
    plot(1:N,rad2deg(eul_alg(:,3)),'m-','LineWidth',1.5);
    xlabel('Frame'); ylabel('Angle (deg)');
    title('Orientation Comparison (GT vs. Algorithm)');
    legend({'GT Roll','Est Roll','GT Pitch','Est Pitch','GT Yaw','Est Yaw'},'Location','best');

    %% (11) Error computation
    pos_error = matched_t_opt - t_tgt_in_cam_alg;
    eul_error = eul_opt - eul_alg;
    deg_error = rad2deg(eul_error);

    figure('Name','Position Error','NumberTitle','off');
    subplot(3,1,1); hold on; grid on;
    plot(1:N, pos_error(:,1),'r-','LineWidth',1.5);
    xlabel('Frame'); ylabel('Err X (m)');
    title('Position Error in X');
    subplot(3,1,2); hold on; grid on;
    plot(1:N, pos_error(:,2),'g-','LineWidth',1.5);
    xlabel('Frame'); ylabel('Err Y (m)');
    title('Position Error in Y');
    subplot(3,1,3); hold on; grid on;
    plot(1:N, pos_error(:,3),'b-','LineWidth',1.5);
    xlabel('Frame'); ylabel('Err Z (m)');
    title('Position Error in Z');

    figure('Name','Orientation Error','NumberTitle','off');
    subplot(3,1,1); hold on; grid on;
    plot(1:N, deg_error(:,1),'r-','LineWidth',1.5);
    xlabel('Frame'); ylabel('Roll Err (deg)');
    title('Orientation Error in Roll');
    subplot(3,1,2); hold on; grid on;
    plot(1:N, deg_error(:,2),'g-','LineWidth',1.5);
    xlabel('Frame'); ylabel('Pitch Err (deg)');
    title('Orientation Error in Pitch');
    subplot(3,1,3); hold on; grid on;
    plot(1:N, deg_error(:,3),'b-','LineWidth',1.5);
    xlabel('Frame'); ylabel('Yaw Err (deg)');
    title('Orientation Error in Yaw');

    disp("Finished plotting. Ensure you actually set 'optTimestamps(i)' using msgList so the time-based matching isn't stuck on one frame!");
end