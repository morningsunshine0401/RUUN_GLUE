%% Script to parse and compare relative poses, then automatically save all figures
% Remove the function signature so this is a script, not a function.

clear all; close all; clc;

% (0) Toggle this to switch between raw data or KF/refined data

useRawData = false;%true;  % If true, use 'object_*' fields; if false, use 'kf_*' fields.
%useRawData = true;

%% (1) Preparations


% This is the one taht wroked well when tested at 20250706

% bagFile = '/media/runbk0401/Storage5/RUUN_GLUE_DATABASE/db3-IMMPORTANT/20250128_test2.db3';
% jsonPath = 'results_20250706_ICUAS_5.json';

% bagFile = '/media/runbk0401/Storage5/RUUN_GLUE_DATABASE/db3-IMMPORTANT/20250128_test2.db3';
% jsonPath = 'pose_results_5.json';

%% 20250902
%bagFile = '/media/runbk0401/Storage5/RUUN_GLUE_DATABASE/db3-IMMPORTANT/20250128_test5.db3';
%%jsonPath = '20250706_ICUAS_2.json';
%jsonPath = 'rtmpose_aircraft_pose_results_57_ICUAS_Blur.json';

% 20250902
bagFile = '/media/runbk0401/Storage5/RUUN_GLUE_DATABASE/db3-IMMPORTANT/20250128_test2.db3';
%jsonPath = '20250706_ICUAS_2.json';
jsonPath = 'VAPE_MK53_results_6.json';

% Extract the test case name from bagFile or jsonPath
[~, bagBase, ~] = fileparts(bagFile);
[~, jsonBase, ~] = fileparts(jsonPath);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% (A) Create a "Matlab_results" folder and a subfolder with today's date to save plots
baseFolder = 'Matlab_results';
if ~exist(baseFolder, 'dir')
    mkdir(baseFolder);
end

dateStr = datestr(now,'yyyy-mm-dd');
%dateStr = '2025-02-123'
%saveFolder = fullfile(baseFolder, dateStr);



testCaseFolder = fullfile(baseFolder, dateStr, jsonBase);


% if ~exist(saveFolder, 'dir')
%     mkdir(saveFolder);
% end

% Ensure the test case-specific folder exists
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
        optTimestamps(i) = posixtime(tVal);    % convert datetime -> seconds
    else
        optTimestamps(i) = double(tVal);
    end
end

%% (3b) Plot Raw Ground Truth Trajectories from Bag File
figure('Name','Raw GT Trajectories (World Frame)','NumberTitle','off');
hold on; grid on; axis equal;
title('Raw Ground Truth Trajectories in World Frame');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');

% Plot camera and target trajectories
plot3(cam_pos_world(:,1), cam_pos_world(:,2), cam_pos_world(:,3), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Camera Trajectory');
plot3(tgt_pos_world(:,1), tgt_pos_world(:,2), tgt_pos_world(:,3), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Target Trajectory');

% Mark start and end points
plot3(cam_pos_world(1,1), cam_pos_world(1,2), cam_pos_world(1,3), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 8, 'DisplayName', 'Camera Start');
plot3(tgt_pos_world(1,1), tgt_pos_world(1,2), tgt_pos_world(1,3), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8, 'DisplayName', 'Target Start');

legend('Location', 'best');
hold off;
print(gcf, fullfile(saveFolder, [bagBase '_Raw_GT_Trajectories.png']), '-dpng','-r300');


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

% Decide which fields to use for rotation/translation
if useRawData
    % Use raw data fields
    rotField = 'object_rotation_in_cam';    
    transField = 'object_translation_in_cam';
    dataTypeTag = 'raw';
else
    % % Use refined or KF fields
    % rotField = 'kf_rotation_matrix';        
    % transField = 'kf_translation_vector';
    % dataTypeTag = 'kf';
    % fprintf("used KF!");

    % This is for jsons from VAPE_MK43_JSON
    rotField = 'rotation_matrix';        
    transField = 'position';
    dataTypeTag = 'kf';
    fprintf("used KF!");
end

% for i = 1:N_json
%     %R_arr = algData(i).(rotField);   % This will be 9 floats (3x3) in either case
%     %t_arr = algData(i).(transField); % 3 floats
% 
%     % Debug: Print all available fields in the first element
%     %disp(fieldnames(algData{1}));
% 
%     R_arr = algData{i}.(rotField);   % This will be 9 floats (3x3) in either case
%     t_arr = algData{i}.(transField); % 3 floats
% 
%     % Reshape R_arr into 3x3
%     R_tgt_in_cam_alg{i} = reshape(R_arr,[3,3]);
%     t_tgt_in_cam_alg(i,:) = t_arr;
% 
%     % The algorithm's timestamp for time-based matching
%     %algTimes(i) = algData(i).timestamp;
%     algTimes(i) = algData{i}.timestamp;
% end

% Initialize storage with NaNs to handle missing values
R_tgt_in_cam_alg = cell(N_json, 1);
t_tgt_in_cam_alg = nan(N_json, 3);
algTimes         = nan(N_json, 1);

for i = 1:N_json

    thisEntry = algData{i}; % This is for test 4 
    %thisEntry = algData(i); % this is for test 2

    % Check if required fields exist
    if isfield(thisEntry, rotField) && isfield(thisEntry, transField)
        R_arr = thisEntry.(rotField);
        t_arr = thisEntry.(transField);

        % Check if rotation matrix is valid
        if ~isempty(R_arr) && numel(R_arr) == 9
            R_tgt_in_cam_alg{i} = reshape(R_arr, [3,3]);
        else
            warning("Invalid or missing rotation matrix at index %d. Assigning NaN.", i);
            R_tgt_in_cam_alg{i} = NaN(3,3);
        end

        % Assign translation vector if valid
        if ~isempty(t_arr) && numel(t_arr) == 3
            t_tgt_in_cam_alg(i,:) = t_arr(:)';  
        else
            warning("Invalid or missing translation vector at index %d. Assigning NaN.", i);
            t_tgt_in_cam_alg(i,:) = NaN(1,3);
        end

        % Store timestamp
        algTimes(i) = thisEntry.timestamp;
    else
        warning("Missing fields in JSON entry %d. Assigning NaN.", i);
        R_tgt_in_cam_alg{i} = NaN(3,3);
        t_tgt_in_cam_alg(i,:) = NaN(1,3);
    end
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
% Correctly pre-allocate arrays for the matched ground truth data,
% with a size equal to the number of entries in the JSON file.
matched_t_opt = zeros(N_json, 3);
matched_R_opt = cell(N_json, 1);
valid_gt_indices = zeros(N_json, 1);

% Loop through each entry in the algorithm's data (JSON)
for i = 1:N_json
    % Get the timestamp from the current JSON entry
    thisAlgTime = algTimes(i);
    
    % Find the index of the ground truth entry with the closest timestamp
    [~, idxClosest] = min(abs(optTimestamps - thisAlgTime));
    
    % Store the matched ground truth data
    matched_t_opt(i,:) = t_rel_array(idxClosest,:);
    matched_R_opt{i} = R_rel_array(:,:,idxClosest);
    valid_gt_indices(i) = idxClosest;
end

% For plotting, we should use the matched ground truth data.
% The number of points in the plots will now correctly correspond
% to the number of entries in the JSON file, but the GT data itself
% is now properly synchronized.
N = N_json;




% %% (8) Plot 3D scatter: GT vs. Algorithm
% figure('Name','3D Relative Poses','NumberTitle','off');
% hold on; grid on; axis equal;
% xlabel('X (m)','FontSize',16); ylabel('Y (m)','FontSize',16); zlabel('Z (m)','FontSize',16);
% title('3D Relative Poses (Time-Synced)','FontSize',18);
% 
% scale = 2.0;
% for i = 1:N_json
%     % GT
%     plot3(matched_t_opt(i,1), matched_t_opt(i,2), matched_t_opt(i,3), ...
%           'o','Color',[0,0.447,0.741],'MarkerFaceColor',[0,0.447,0.741]);
%     R_opt = matched_R_opt{i};
%     quiver3(matched_t_opt(i,1), matched_t_opt(i,2), matched_t_opt(i,3), ...
%             scale*R_opt(1,1),scale*R_opt(2,1),scale*R_opt(3,1), 'Color',[1,1,0]);
% 
%     % Algorithm
%     plot3(t_tgt_in_cam_alg(i,1), t_tgt_in_cam_alg(i,2), t_tgt_in_cam_alg(i,3), ...
%           'x','Color',[0.85,0.325,0.098],'MarkerSize',8,'LineWidth',1.5);
%     R_alg = R_tgt_in_cam_alg{i};
%     quiver3(t_tgt_in_cam_alg(i,1), t_tgt_in_cam_alg(i,2), t_tgt_in_cam_alg(i,3), ...
%             scale*R_alg(1,1),scale*R_alg(2,1),scale*R_alg(3,1), 'Color',[1,0,0]);
% end
% legend({'GT Target','GT Orientation','Estimated Target','Est Orientation'});
% hold off;
% 
% % Automatically save figure (note the dataTypeTag)
% print(gcf, fullfile(saveFolder, ...
%     [bagBase '_' jsonBase '_' dataTypeTag '_3DRelativePoses.png']), '-dpng','-r300');

%% (9) Plot X/Y/Z comparison
figure('Name','Position Comparison','NumberTitle','off');
hold on; grid on;
N = N_json;
plot(1:N, matched_t_opt(:,1),'b--','LineWidth',1.7);
plot(1:N, t_tgt_in_cam_alg(:,1),'b-','LineWidth',1.7);
plot(1:N, matched_t_opt(:,2),'g--','LineWidth',1.7);
plot(1:N, t_tgt_in_cam_alg(:,2),'g-','LineWidth',1.7);
plot(1:N, matched_t_opt(:,3),'r--','LineWidth',1.7);
plot(1:N, t_tgt_in_cam_alg(:,3),'r-','LineWidth',1.7);
set(gca, 'FontSize', 14);
xlabel('Frame','FontSize',16); ylabel('Relative Position (m)','FontSize',16);
legend({'GT X','Est X','GT Y','Est Y','GT Z','Est Z'},'Location','best');
hold off;

% Automatically save figure
print(gcf, fullfile(saveFolder, ...
    [bagBase '_' jsonBase '_' dataTypeTag '_PositionComparison.png']), '-dpng','-r300');

%% (10) Orientation comparison
eul_opt = zeros(N,3);
eul_alg = zeros(N,3);
for i = 1:N
    eul_opt(i,:) = rotm2eul(matched_R_opt{i}, 'XYZ');
    eul_alg(i,:) = rotm2eul(R_tgt_in_cam_alg{i}, 'XYZ');
end

figure('Name','Orientation Comparison','NumberTitle','off');
hold on; grid on;
plot(1:N,rad2deg(eul_opt(:,1)),'b--','LineWidth',1.7);
plot(1:N,rad2deg(eul_alg(:,1)),'b-','LineWidth',1.7);
plot(1:N,rad2deg(eul_opt(:,2)),'g--','LineWidth',1.7);
plot(1:N,rad2deg(eul_alg(:,2)),'g-','LineWidth',1.7);
plot(1:N,rad2deg(eul_opt(:,3)),'m--','LineWidth',1.7);
plot(1:N,rad2deg(eul_alg(:,3)),'m-','LineWidth',1.7);
set(gca, 'FontSize', 14);
xlabel('Frame','FontSize',16); ylabel('Angle (deg)','FontSize',16);
legend({'GT Roll','Est Roll','GT Pitch','Est Pitch','GT Yaw','Est Yaw'},'Location','best');
hold off;

% Automatically save figure
print(gcf, fullfile(saveFolder, ...
    [bagBase '_' jsonBase '_' dataTypeTag '_OrientationComparison.png']), '-dpng','-r300');

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
% Automatically save figure
print(gcf, fullfile(saveFolder, ...
    [bagBase '_' jsonBase '_' dataTypeTag '_PositionError.png']), '-dpng','-r300');

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
% Automatically save figure
print(gcf, fullfile(saveFolder, ...
    [bagBase '_' jsonBase '_' dataTypeTag '_OrientationError.png']), '-dpng','-r300');

%% (12) (Optional) Distance comparison if desired
%{
gt_distances  = vecnorm(matched_t_opt, 2, 2); 
est_distances = vecnorm(t_tgt_in_cam_alg, 2, 2);
distance_error = abs(gt_distances - est_distances);

figure('Name', 'Distance Comparison', 'NumberTitle', 'off');
hold on; grid on;
plot(1:N_json, gt_distances, 'b--', 'LineWidth', 1.5);
plot(1:N_json, est_distances, 'r-', 'LineWidth', 1.5);
plot(1:N_json, distance_error, 'k-.', 'LineWidth', 1.5);
xlabel('Frame');
ylabel('Distance (m)');
title('Distance Comparison (GT vs. Algorithm)');
legend({'GT Distance', 'Est Distance', 'Distance Error'}, 'Location', 'best');
hold off;

print(gcf, fullfile(saveFolder, ...
    [bagBase '_' jsonBase '_' dataTypeTag '_DistanceComparison.png']), '-dpng','-r300');
%}

disp("Finished plotting. All figures saved automatically in: " + saveFolder);
