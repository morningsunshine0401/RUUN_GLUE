%% Script to parse and compare relative poses, then automatically save all figures
% Remove the function signature so this is a script, not a function.

clear all; close all; clc;

%% (0) Toggle this to switch between raw data or KF/refined data

useRawData = false;  % If true, use 'object_*' fields; if false, use 'kf_*' fields.
%useRawData = true; 

%% (1) Preparations
% 
% % Edit these paths to match your data:
% bagFile  = '20250128_test4.db3';
% jsonPath = 'pose_estimation.json';

bagFile = '/media/runbk0401/Storage5/RUUN_GLUE_DATABASE/db3-IMMPORTANT/20250128_test2.db3';
jsonPath = 'results_20250706_ICUAS_10.json';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% (A) Create a "Matlab_results" folder and a subfolder with today's date to save plots
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
        optTimestamps(i) = posixtime(tVal);  % convert datetime -> seconds
    else
        optTimestamps(i) = double(tVal);
    end
end

%% (4) Convert quaternions -> rotation matrices
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

%% (x) Frame selection for plotting
% Specify the indices of frames you want to plot.
% For example, to plot frames 1 to 40:
% Specify the desired number of frames to plot
desiredFrameCount = 600;
% Automatically select frames from 1 up to the lesser of desiredFrameCount or the available JSON frames
selectedFrames = 1:min(desiredFrameCount, N_json);
if max(selectedFrames) > N_json
    error('Selected frames exceed the available JSON frames.');
end

% We'll store the rotation and translation from the JSON
R_tgt_in_cam_alg = cell(N_json,1);
t_tgt_in_cam_alg = nan(N_json,3);
algTimes         = nan(N_json,1);

% Decide which fields to use for rotation/translation
if useRawData
    % Use raw data fields
    rotField = 'object_rotation_in_cam';
    transField = 'object_translation_in_cam';
    dataTypeTag = 'raw';
    fprintf("Using raw data fields: object_rotation_in_cam, object_translation_in_cam\n");
else
    % % Use refined or KF fields
    % rotField = 'kf_rotation_matrix';
    % transField = 'kf_translation_vector';
    % dataTypeTag = 'kf';
    % fprintf("Using KF data fields: kf_rotation_matrix, kf_translation_vector\n");

    % This is for jsons from VAPE_MK43_JSON
    rotField = 'rotation_matrix';        
    transField = 'position';
    dataTypeTag = 'kf';
    fprintf("Using KF data fields: kf_rotation_matrix, kf_translation_vector\n");
end

for i = 1:N_json
    thisEntry = algData{i};

    % Check if required fields exist
    if isfield(thisEntry, rotField) && isfield(thisEntry, transField)
        R_arr = thisEntry.(rotField);
        t_arr = thisEntry.(transField);

        % Reshape R_arr into 3x3 if valid
        if ~isempty(R_arr) && numel(R_arr) == 9
            R_tgt_in_cam_alg{i} = reshape(R_arr, [3,3]);
        else
            warning("Invalid or missing rotation matrix at index %d. Using NaN.", i);
            R_tgt_in_cam_alg{i} = NaN(3,3);
        end

        % Assign translation vector if valid
        if ~isempty(t_arr) && numel(t_arr) == 3
            t_tgt_in_cam_alg(i,:) = t_arr(:)';
        else
            warning("Invalid or missing translation vector at index %d. Using NaN.", i);
            t_tgt_in_cam_alg(i,:) = NaN(1,3);
        end

        % Store timestamp
        if isfield(thisEntry, 'timestamp')
            algTimes(i) = thisEntry.timestamp;
        else
            algTimes(i) = NaN;
        end
    else
        warning("Missing fields in JSON entry %d. Assigning NaN.", i);
        R_tgt_in_cam_alg{i} = NaN(3,3);
        t_tgt_in_cam_alg(i,:) = NaN(1,3);
    end
end

%% (6) Optionally align the coordinate system
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

% For naming saved figures
[~, bagBase, ~] = fileparts(bagFile);
[~, jsonBase, ~] = fileparts(jsonPath);

%% (9) Plot X/Y/Z comparison for selected frames
figure('Name','Position Comparison','NumberTitle','off');
hold on; grid on;
plot(selectedFrames, matched_t_opt(selectedFrames,1),'b--','LineWidth',1.7);
plot(selectedFrames, t_tgt_in_cam_alg(selectedFrames,1),'b-','LineWidth',1.7);
plot(selectedFrames, matched_t_opt(selectedFrames,2),'g--','LineWidth',1.7);
plot(selectedFrames, t_tgt_in_cam_alg(selectedFrames,2),'g-','LineWidth',1.7);
plot(selectedFrames, matched_t_opt(selectedFrames,3),'r--','LineWidth',1.7);
plot(selectedFrames, t_tgt_in_cam_alg(selectedFrames,3),'r-','LineWidth',1.7);
set(gca, 'FontSize', 14);
xlabel('Frame','FontSize',16); ylabel('Relative Position (m)','FontSize',16);
legend({'GT X','Est X','GT Y','Est Y','GT Z','Est Z'},'Location','best');
hold off;

print(gcf, fullfile(saveFolder, ...
    [bagBase '_' jsonBase '_' dataTypeTag '_PositionComparison.png']), '-dpng','-r300');

%% (10) Orientation comparison for selected frames
% Compute Euler angles for all frames first
eul_all = zeros(N_json,3);
eul_alg_all = zeros(N_json,3);
for i = 1:N_json
    eul_all(i,:) = rotm2eul(matched_R_opt{i}, 'XYZ');
    eul_alg_all(i,:) = rotm2eul(R_tgt_in_cam_alg{i}, 'XYZ');
end

% Restrict to selected frames
eul_opt = eul_all(selectedFrames, :);
eul_alg = eul_alg_all(selectedFrames, :);

figure('Name','Orientation Comparison','NumberTitle','off');
hold on; grid on;
plot(selectedFrames, rad2deg(eul_opt(:,1)),'b--','LineWidth',1.7);
plot(selectedFrames, rad2deg(eul_alg(:,1)),'b-','LineWidth',1.7);
plot(selectedFrames, rad2deg(eul_opt(:,2)),'g--','LineWidth',1.7);
plot(selectedFrames, rad2deg(eul_alg(:,2)),'g-','LineWidth',1.7);
plot(selectedFrames, rad2deg(eul_opt(:,3)),'m--','LineWidth',1.7);
plot(selectedFrames, rad2deg(eul_alg(:,3)),'m-','LineWidth',1.7);
set(gca, 'FontSize', 14);
xlabel('Frame','FontSize',16); ylabel('Angle (deg)','FontSize',16);
legend({'GT Roll','Est Roll','GT Pitch','Est Pitch','GT Yaw','Est Yaw'},'Location','best');
hold off;

print(gcf, fullfile(saveFolder, ...
    [bagBase '_' jsonBase '_' dataTypeTag '_OrientationComparison.png']), '-dpng','-r300');

%% (11) Error computation for selected frames
pos_error = matched_t_opt - t_tgt_in_cam_alg;
eul_error = eul_all - eul_alg_all;
deg_error = rad2deg(eul_error);

% Restrict errors to selected frames
pos_error_plot = pos_error(selectedFrames,:);
deg_error_plot = deg_error(selectedFrames,:);

figure('Name','Position Error','NumberTitle','off');
subplot(3,1,1); hold on; grid on;
plot(selectedFrames, pos_error_plot(:,1),'r-','LineWidth',1.5);
xlabel('Frame'); ylabel('Err X (m)');
title('Position Error in X');
subplot(3,1,2); hold on; grid on;
plot(selectedFrames, pos_error_plot(:,2),'g-','LineWidth',1.5);
xlabel('Frame'); ylabel('Err Y (m)');
title('Position Error in Y');
subplot(3,1,3); hold on; grid on;
plot(selectedFrames, pos_error_plot(:,3),'b-','LineWidth',1.5);
xlabel('Frame'); ylabel('Err Z (m)');
title('Position Error in Z');
print(gcf, fullfile(saveFolder, ...
    [bagBase '_' jsonBase '_' dataTypeTag '_PositionError.png']), '-dpng','-r300');

figure('Name','Orientation Error','NumberTitle','off');
subplot(3,1,1); hold on; grid on;
plot(selectedFrames, deg_error_plot(:,1),'r-','LineWidth',1.5);
xlabel('Frame'); ylabel('Roll Err (deg)');
title('Orientation Error in Roll');
subplot(3,1,2); hold on; grid on;
plot(selectedFrames, deg_error_plot(:,2),'g-','LineWidth',1.5);
xlabel('Frame'); ylabel('Pitch Err (deg)');
title('Orientation Error in Pitch');
subplot(3,1,3); hold on; grid on;
plot(selectedFrames, deg_error_plot(:,3),'b-','LineWidth',1.5);
xlabel('Frame'); ylabel('Yaw Err (deg)');
title('Orientation Error in Yaw');
print(gcf, fullfile(saveFolder, ...
    [bagBase '_' jsonBase '_' dataTypeTag '_OrientationError.png']), '-dpng','-r300');

%% (12) (Optional) Distance comparison if desired for selected frames
%{
gt_distances  = vecnorm(matched_t_opt, 2, 2);
est_distances = vecnorm(t_tgt_in_cam_alg, 2, 2);
distance_error = abs(gt_distances - est_distances);

figure('Name', 'Distance Comparison', 'NumberTitle', 'off');
hold on; grid on;
plot(selectedFrames, gt_distances(selectedFrames), 'b--', 'LineWidth', 1.5);
plot(selectedFrames, est_distances(selectedFrames), 'r-', 'LineWidth', 1.5);
plot(selectedFrames, distance_error(selectedFrames), 'k-.', 'LineWidth', 1.5);
xlabel('Frame');
ylabel('Distance (m)');
title('Distance Comparison (GT vs. Algorithm)');
legend({'GT Distance', 'Est Distance', 'Distance Error'}, 'Location', 'best');
hold off;

print(gcf, fullfile(saveFolder, ...
    [bagBase '_' jsonBase '_' dataTypeTag '_DistanceComparison.png']), '-dpng','-r300');
%}

disp("Finished plotting. All figures saved automatically in: " + saveFolder);







% %% Script to parse and compare relative poses, then automatically save all figures
% % Remove the function signature so this is a script, not a function.
% 
% clear all; close all; clc;
% 
% 
% %% (0) Toggle this to switch between raw data or KF/refined data
% 
% %useRawData = false;  % If true, use 'object_*' fields; if false, use 'kf_*' fields.
% 
% useRawData = true; 
% 
% %% (1) Preparations
% 
% % Edit these paths to match your data:
% bagFile  = '20250128_test4.db3';
% jsonPath = '20250311_20250128_test4_thread_5.json';
% 
% %bagFile  = '20250203_test2.db3';
% %jsonPath = '202503011_20250203_test2_pixel.json';
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % (A) Create a "Matlab_results" folder and a subfolder with today's date to save plots
% baseFolder = 'Matlab_results';
% if ~exist(baseFolder, 'dir')
%     mkdir(baseFolder);
% end
% 
% dateStr = datestr(now,'yyyy-mm-dd');
% testCaseFolder = fullfile(baseFolder, dateStr);
% 
% % Extract part of the file name for your test case
% [~, jsonBase, ~] = fileparts(jsonPath);
% testCaseFolder = fullfile(testCaseFolder, jsonBase);
% 
% if ~exist(testCaseFolder, 'dir')
%     mkdir(testCaseFolder);
% end
% 
% saveFolder = testCaseFolder;
% 
% % Our ground-truth topic in the bag
% topicName = "/OPTI/rb_infos";
% 
% %% (2) Read the bag in chronological order
% bagReader = ros2bagreader(bagFile);
% msgList   = bagReader.MessageList;   % Table: {Time, Topic, MessageType}
% allMsgs   = readMessages(bagReader);
% 
% % Find the row indices for our GT topic
% idxOpt = find(msgList.Topic == topicName);
% N_opt  = numel(idxOpt);
% if N_opt < 1
%     error("No messages found on topic '%s' in bag '%s'", topicName, bagFile);
% end
% fprintf("Found %d messages on %s\n", N_opt, topicName);
% 
% %% Allocate storage
% cam_pos_world  = zeros(N_opt,3);
% cam_quat_world = zeros(N_opt,4);
% tgt_pos_world  = zeros(N_opt,3);
% tgt_quat_world = zeros(N_opt,4);
% 
% % We'll store numeric timestamps for each GT message
% optTimestamps = zeros(N_opt,1);
% 
% %% (3) Parse GT data + timestamps
% for i = 1:N_opt
%     row = idxOpt(i);
%     d   = allMsgs{row}.data;  % float[14]: [cam_pos,cam_quat,tgt_pos,tgt_quat]
% 
%     cam_pos_world(i,:)   = d(1:3);
%     cam_quat_world(i,:)  = d(4:7);
%     tgt_pos_world(i,:)   = d(8:10);
%     tgt_quat_world(i,:)  = d(11:14);
% 
%     % Fill the numeric timestamp
%     tVal = msgList.Time(row);
%     if isdatetime(tVal)
%         optTimestamps(i) = posixtime(tVal);  % convert datetime -> seconds
%     else
%         optTimestamps(i) = double(tVal);
%     end
% end
% 
% %% (4) Convert quaternions -> rotation matrices
% R_cam_in_world = zeros(3,3,N_opt);
% R_tgt_in_world = zeros(3,3,N_opt);
% 
% for i = 1:N_opt
%     R_cam_in_world(:,:,i) = quat2rotm(cam_quat_world(i,:));
%     R_tgt_in_world(:,:,i) = quat2rotm(tgt_quat_world(i,:));
% end
% 
% %% (5) Load the JSON pose estimation
% jsonText = fileread(jsonPath);
% algData  = jsondecode(jsonText);
% N_json   = numel(algData);
% 
% %% (x) Frame selection for plotting
% % Specify the indices of frames you want to plot.
% % For example, to plot frames 1 to 40:
% % Specify the desired number of frames to plot
% desiredFrameCount = 400;
% % Automatically select frames from 1 up to the lesser of desiredFrameCount or the available JSON frames
% selectedFrames = 1:min(desiredFrameCount, N_json);
% if max(selectedFrames) > N_json
%     error('Selected frames exceed the available JSON frames.');
% end
% 
% % We'll store the rotation and translation from the JSON
% R_tgt_in_cam_alg = cell(N_json,1);
% t_tgt_in_cam_alg = nan(N_json,3);
% algTimes         = nan(N_json,1);
% 
% % Decide which fields to use for rotation/translation
% if useRawData
%     % Use raw data fields
%     rotField = 'object_rotation_in_cam';
%     transField = 'object_translation_in_cam';
%     dataTypeTag = 'raw';
%     fprintf("Using raw data fields: object_rotation_in_cam, object_translation_in_cam\n");
% else
%     % Use refined or KF fields
%     rotField = 'kf_rotation_matrix';
%     transField = 'kf_translation_vector';
%     dataTypeTag = 'kf';
%     fprintf("Using KF data fields: kf_rotation_matrix, kf_translation_vector\n");
% end
% 
% for i = 1:N_json
% 
%     % Test 4 is
%     thisEntry = algData{i};
% 
%     % Test 2 is
%     %thisEntry = algData(i);
% 
%     % Check if required fields exist
%     if isfield(thisEntry, rotField) && isfield(thisEntry, transField)
%         R_arr = thisEntry.(rotField);
%         t_arr = thisEntry.(transField);
% 
%         % Reshape R_arr into 3x3 if valid
%         if ~isempty(R_arr) && numel(R_arr) == 9
%             R_tgt_in_cam_alg{i} = reshape(R_arr, [3,3]);
%         else
%             warning("Invalid or missing rotation matrix at index %d. Using NaN.", i);
%             R_tgt_in_cam_alg{i} = NaN(3,3);
%         end
% 
%         % Assign translation vector if valid
%         if ~isempty(t_arr) && numel(t_arr) == 3
%             t_tgt_in_cam_alg(i,:) = t_arr(:)';
%         else
%             warning("Invalid or missing translation vector at index %d. Using NaN.", i);
%             t_tgt_in_cam_alg(i,:) = NaN(1,3);
%         end
% 
%         % Store timestamp
%         if isfield(thisEntry, 'timestamp')
%             algTimes(i) = thisEntry.timestamp;
%         else
%             algTimes(i) = NaN;
%         end
%     else
%         warning("Missing fields in JSON entry %d. Assigning NaN.", i);
%         R_tgt_in_cam_alg{i} = NaN(3,3);
%         t_tgt_in_cam_alg(i,:) = NaN(1,3);
%     end
% end
% 
% %% (6) Optionally align the coordinate system
% % Example: R_alignment flips X and Y, keeps Z
% R_alignment = [-1, 0,  0;
%                0, -1,  0;
%                0,  0,  1];
% 
% t_rel_array = zeros(N_opt,3);
% R_rel_array = zeros(3,3,N_opt);
% 
% for i = 1:N_opt
%     % Align camera rotation
%     R_cam_aligned = R_alignment * R_cam_in_world(:,:,i) * R_alignment';
%     % Align target rotation
%     R_tgt_aligned = R_alignment * R_tgt_in_world(:,:,i) * R_alignment';
% 
%     % Relative rotation: camera->target
%     R_rel_array(:,:,i) = R_cam_aligned' * R_tgt_aligned;
% 
%     % Align positions
%     cam_pos_aligned = (R_alignment * cam_pos_world(i,:).')';
%     tgt_pos_aligned = (R_alignment * tgt_pos_world(i,:).')';
% 
%     % Relative position in camera frame
%     t_rel_array(i,:) = (R_cam_aligned' * (tgt_pos_aligned - cam_pos_aligned).').';
% end
% 
% %% (7) Time-based matching
% matched_t_opt = zeros(N_json,3);
% matched_R_opt = cell(N_json,1);
% 
% for i = 1:N_json
%     thisAlgTime = algTimes(i);
%     [~, idxClosest] = min(abs(optTimestamps - thisAlgTime));
% 
%     matched_t_opt(i,:)  = t_rel_array(idxClosest,:);
%     matched_R_opt{i}    = R_rel_array(:,:,idxClosest);
% end
% 
% % For naming saved figures
% [~, bagBase, ~] = fileparts(bagFile);
% [~, jsonBase, ~] = fileparts(jsonPath);
% 
% %% (9) Plot X/Y/Z comparison for selected frames
% figure('Name','Position Comparison','NumberTitle','off');
% hold on; grid on;
% plot(selectedFrames, matched_t_opt(selectedFrames,1),'b--','LineWidth',1.7);
% plot(selectedFrames, t_tgt_in_cam_alg(selectedFrames,1),'b-','LineWidth',1.7);
% plot(selectedFrames, matched_t_opt(selectedFrames,2),'g--','LineWidth',1.7);
% plot(selectedFrames, t_tgt_in_cam_alg(selectedFrames,2),'g-','LineWidth',1.7);
% plot(selectedFrames, matched_t_opt(selectedFrames,3),'r--','LineWidth',1.7);
% plot(selectedFrames, t_tgt_in_cam_alg(selectedFrames,3),'r-','LineWidth',1.7);
% set(gca, 'FontSize', 14);
% xlabel('Frame','FontSize',16); ylabel('Relative Position (m)','FontSize',16);
% legend({'GT X','Est X','GT Y','Est Y','GT Z','Est Z'},'Location','best');
% hold off;
% 
% print(gcf, fullfile(saveFolder, ...
%     [bagBase '_' jsonBase '_' dataTypeTag '_PositionComparison.png']), '-dpng','-r300');
% 
% %% (10) Orientation comparison for selected frames
% % Compute Euler angles for all frames first
% eul_all = zeros(N_json,3);
% eul_alg_all = zeros(N_json,3);
% for i = 1:N_json
%     eul_all(i,:) = rotm2eul(matched_R_opt{i}, 'XYZ');
%     eul_alg_all(i,:) = rotm2eul(R_tgt_in_cam_alg{i}, 'XYZ');
% end
% 
% % Restrict to selected frames
% eul_opt = eul_all(selectedFrames, :);
% eul_alg = eul_alg_all(selectedFrames, :);
% 
% figure('Name','Orientation Comparison','NumberTitle','off');
% hold on; grid on;
% plot(selectedFrames, rad2deg(eul_opt(:,1)),'b--','LineWidth',1.7);
% plot(selectedFrames, rad2deg(eul_alg(:,1)),'b-','LineWidth',1.7);
% plot(selectedFrames, rad2deg(eul_opt(:,2)),'g--','LineWidth',1.7);
% plot(selectedFrames, rad2deg(eul_alg(:,2)),'g-','LineWidth',1.7);
% plot(selectedFrames, rad2deg(eul_opt(:,3)),'m--','LineWidth',1.7);
% plot(selectedFrames, rad2deg(eul_alg(:,3)),'m-','LineWidth',1.7);
% set(gca, 'FontSize', 14);
% xlabel('Frame','FontSize',16); ylabel('Angle (deg)','FontSize',16);
% legend({'GT Roll','Est Roll','GT Pitch','Est Pitch','GT Yaw','Est Yaw'},'Location','best');
% hold off;
% 
% print(gcf, fullfile(saveFolder, ...
%     [bagBase '_' jsonBase '_' dataTypeTag '_OrientationComparison.png']), '-dpng','-r300');
% 
% %% (11) Error computation for selected frames
% pos_error = matched_t_opt - t_tgt_in_cam_alg;
% eul_error = eul_all - eul_alg_all;
% deg_error = rad2deg(eul_error);
% 
% % Restrict errors to selected frames
% pos_error_plot = pos_error(selectedFrames,:);
% deg_error_plot = deg_error(selectedFrames,:);
% 
% figure('Name','Position Error','NumberTitle','off');
% subplot(3,1,1); hold on; grid on;
% plot(selectedFrames, pos_error_plot(:,1),'r-','LineWidth',1.5);
% xlabel('Frame'); ylabel('Err X (m)');
% title('Position Error in X');
% subplot(3,1,2); hold on; grid on;
% plot(selectedFrames, pos_error_plot(:,2),'g-','LineWidth',1.5);
% xlabel('Frame'); ylabel('Err Y (m)');
% title('Position Error in Y');
% subplot(3,1,3); hold on; grid on;
% plot(selectedFrames, pos_error_plot(:,3),'b-','LineWidth',1.5);
% xlabel('Frame'); ylabel('Err Z (m)');
% title('Position Error in Z');
% print(gcf, fullfile(saveFolder, ...
%     [bagBase '_' jsonBase '_' dataTypeTag '_PositionError.png']), '-dpng','-r300');
% 
% figure('Name','Orientation Error','NumberTitle','off');
% subplot(3,1,1); hold on; grid on;
% % 
% % % sd edit
% % temp = [];
% % for i = 1:length(deg_error_plot)
% %     if (abs(deg_error_plot(i,1)) < 20 && abs(deg_error_plot(i,2)) < 20 &&  abs(deg_error_plot(i,3)) < 20)
% %         temp = [temp; i];
% %     end
% % end
% 
% plot(1:length(temp), deg_error_plot(temp,1),'r-','LineWidth',1.5);
% xlabel('Frame'); ylabel('Roll Err (deg)');
% title('Orientation Error in Roll');
% subplot(3,1,2); hold on; grid on;
% plot(selectedFrames, deg_error_plot(temp,2),'g-','LineWidth',1.5);
% xlabel('Frame'); ylabel('Pitch Err (deg)');
% title('Orientation Error in Pitch');
% subplot(3,1,3); hold on; grid on;
% plot(selectedFrames, deg_error_plot(temp,3),'b-','LineWidth',1.5);
% xlabel('Frame'); ylabel('Yaw Err (deg)');
% title('Orientation Error in Yaw');
% print(gcf, fullfile(saveFolder, ...
%     [bagBase '_' jsonBase '_' dataTypeTag '_OrientationError.png']), '-dpng','-r300');
% 
% %% (12) (Optional) Distance comparison if desired for selected frames
% %{
% gt_distances  = vecnorm(matched_t_opt, 2, 2);
% est_distances = vecnorm(t_tgt_in_cam_alg, 2, 2);
% distance_error = abs(gt_distances - est_distances);
% 
% figure('Name', 'Distance Comparison', 'NumberTitle', 'off');
% hold on; grid on;
% plot(selectedFrames, gt_distances(selectedFrames), 'b--', 'LineWidth', 1.5);
% plot(selectedFrames, est_distances(selectedFrames), 'r-', 'LineWidth', 1.5);
% plot(selectedFrames, distance_error(selectedFrames), 'k-.', 'LineWidth', 1.5);
% xlabel('Frame');
% ylabel('Distance (m)');
% title('Distance Comparison (GT vs. Algorithm)');
% legend({'GT Distance', 'Est Distance', 'Distance Error'}, 'Location', 'best');
% hold off;
% 
% print(gcf, fullfile(saveFolder, ...
%     [bagBase '_' jsonBase '_' dataTypeTag '_DistanceComparison.png']), '-dpng','-r300');
% %}
% 
% disp("Finished plotting. All figures saved automatically in: " + saveFolder);
