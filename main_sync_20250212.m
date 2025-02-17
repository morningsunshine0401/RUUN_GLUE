%% Script to parse and compare relative poses, then automatically save all figures
% Remove the function signature so this is a script, not a function.

clear all; close all; clc;

% (0) Toggle this to switch between raw data or KF/refined data

%useRawData = false;%true;  % If true, use 'object_*' fields; if false, use 'kf_*' fields.
useRawData = true;

% (A) Create a "Matlab_results" folder and a subfolder with today's date to save plots
baseFolder = 'Matlab_results';
if ~exist(baseFolder, 'dir')
    mkdir(baseFolder);
end

dateStr = datestr(now,'yyyy-mm-dd');
saveFolder = fullfile(baseFolder, dateStr);
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

%% (1) Preparations
% Edit these paths to match your data:

bagFile  = '20250128_test4.db3';
jsonPath = '20250212_test4_adaptive2_3.json';

%bagFile  = '20250120_test5.db3';
%jsonPath = '20250212_test5_adaptive2_5.json';

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

%% (5) Load the JSON pose estimation and extract additional metrics
jsonText = fileread(jsonPath);
algData  = jsondecode(jsonText);
N_json   = numel(algData);

R_tgt_in_cam_alg = cell(N_json,1);
t_tgt_in_cam_alg = zeros(N_json,3);
algTimes         = zeros(N_json,1);

% Allocate storage for additional algorithm metrics
%mconf          = zeros(N_json, 1);
coverage_score = zeros(N_json, 1);
skip_count     = zeros(N_json, 1);
mahalanobis_sq = zeros(N_json, 1);

% Decide which fields to use for rotation/translation
if useRawData
    % Use raw data fields
    rotField = 'object_rotation_in_cam';    
    transField = 'object_translation_in_cam';
    dataTypeTag = 'raw';
else
    % Use refined or KF fields
    rotField = 'kf_rotation_matrix';        
    transField = 'kf_translation_vector';
    dataTypeTag = 'kf';
end

for i = 1:N_json
    % Extract rotation and translation from the JSON
    R_arr = algData(i).(rotField);   % This will be 9 floats (3x3) in either case
    t_arr = algData(i).(transField);   % 3 floats

    % Reshape R_arr into 3x3 and store
    R_tgt_in_cam_alg{i} = reshape(R_arr,[3,3]);
    t_tgt_in_cam_alg(i,:) = t_arr;

    % The algorithm's timestamp for time-based matching
    algTimes(i) = algData(i).timestamp;
    
    % Extract additional metrics
    %mconf(i)          = algData(i).mconf;
    coverage_score(i) = algData(i).coverage_score;
    skip_count(i)     = algData(i).skip_count;
    mahalanobis_sq(i) = algData(i).mahalanobis_sq;
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

%% (B) For naming saved figures
% Extract the file name portions (without extensions) of the bag and JSON
[~, bagBase, ~] = fileparts(bagFile);
[~, jsonBase, ~] = fileparts(jsonPath);

%% (8) (Optional) 3D scatter plot: GT vs. Algorithm
%{
figure('Name','3D Relative Poses','NumberTitle','off');
hold on; grid on; axis equal;
xlabel('X (m)','FontSize',16); ylabel('Y (m)','FontSize',16); zlabel('Z (m)','FontSize',16);
title('3D Relative Poses (Time-Synced)','FontSize',18);

scale = 2.0;
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

% Automatically save figure (note the dataTypeTag)
print(gcf, fullfile(saveFolder, ...
    [bagBase '_' jsonBase '_' dataTypeTag '_3DRelativePoses.png']), '-dpng','-r300');
%}

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

%% (13) Plot additional algorithm metrics
% % Plot 'mconf'
% figure('Name','Algorithm Metric: mconf','NumberTitle','off');
% plot(1:N_json, mconf, 'k-', 'LineWidth',1.7);
% grid on; set(gca, 'FontSize', 14);
% xlabel('Frame','FontSize',16); ylabel('mconf','FontSize',16);
% title('Algorithm mconf over Frames');
% print(gcf, fullfile(saveFolder, [bagBase '_' jsonBase '_' dataTypeTag '_mconf.png']), '-dpng','-r300');

% Plot 'coverage_score'
figure('Name','Algorithm Metric: coverage_score','NumberTitle','off');
plot(1:N_json, coverage_score, 'b-', 'LineWidth',1.7);
grid on; set(gca, 'FontSize', 14);
xlabel('Frame','FontSize',16); ylabel('Coverage Score','FontSize',16);
title('Algorithm Coverage Score over Frames');
print(gcf, fullfile(saveFolder, [bagBase '_' jsonBase '_' dataTypeTag '_coverage_score.png']), '-dpng','-r300');

% Plot 'skip_count'
figure('Name','Algorithm Metric: skip_count','NumberTitle','off');
plot(1:N_json, skip_count, 'r-', 'LineWidth',1.7);
grid on; set(gca, 'FontSize', 14);
xlabel('Frame','FontSize',16); ylabel('Skip Count','FontSize',16);
title('Algorithm Skip Count over Frames');
print(gcf, fullfile(saveFolder, [bagBase '_' jsonBase '_' dataTypeTag '_skip_count.png']), '-dpng','-r300');

% Plot 'mahalanobis_sq'
figure('Name','Algorithm Metric: mahalanobis_sq','NumberTitle','off');
plot(1:N_json, mahalanobis_sq, 'm-', 'LineWidth',1.7);
grid on; set(gca, 'FontSize', 14);
xlabel('Frame','FontSize',16); ylabel('Mahalanobis Squared','FontSize',16);
title('Algorithm Mahalanobis Squared over Frames');
print(gcf, fullfile(saveFolder, [bagBase '_' jsonBase '_' dataTypeTag '_mahalanobis_sq.png']), '-dpng','-r300');

disp("Finished plotting. All figures saved automatically in: " + saveFolder);
