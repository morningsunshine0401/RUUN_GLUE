%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example: Convert Target Positions into the "Initial Camera" Frame,
%          referencing how we load data from a ROS2 bag.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

figure_on = 1;
rad2deg   = 180/pi;

% ------------------------------------------------------------------------
% (1) Load the ROS2 bag data (as you already do).
%     We assume you get:
%       cam_pos(i,:)   = [camX, camY, camZ] in world coords (meters)
%       cam_quat(i,:)  = [w, x, y, z] for camera orientation in world frame
%       target_pos(i,:)= [tgtX, tgtY, tgtZ] in world coords (meters)
%     with length data_len.
% ------------------------------------------------------------------------
folderPath = 'test2.db3';
bagReader = ros2bagreader(folderPath);
baginfo   = ros2("bag","info",folderPath);
msgs      = readMessages(bagReader);

% Select the OptiTrack data
OPTI_rb_infos = select(bagReader,"Topic","/OPTI/rb_infos");
OPTI_rb_infos_Filtered = readMessages(OPTI_rb_infos);

% Number of data frames
data_len = length(OPTI_rb_infos_Filtered);

% The data array will have 14 columns:
% [camX, camY, camZ, camQw, camQx, camQy, camQz,
%  tgtX, tgtY, tgtZ, tgtQw, tgtQx, tgtQy, tgtQz]
OPTI_rb_infos_data = zeros(data_len, 14); 

for i=1:data_len
    OPTI_rb_infos_data(i,:) = cell2mat(OPTI_rb_infos_Filtered(i,1)).data';
end

% (Optional) extract timestamps if needed
OPTI_rb_infos_time = (OPTI_rb_infos.MessageList.Time - ...
                      min(OPTI_rb_infos.MessageList.Time));

% Camera Pose in World
cam_pos  = OPTI_rb_infos_data(:,1:3);    % [X_w, Y_w, Z_w]
cam_quat = OPTI_rb_infos_data(:,4:7);    % [w, x, y, z]

% Target Pose in World
target_pos  = OPTI_rb_infos_data(:,8:10);   % [X_w, Y_w, Z_w]
target_quat = OPTI_rb_infos_data(:,11:14);  % [w, x, y, z]

% For debugging, you can convert camera quaternion to yaw/pitch/roll:
[cam_yaw, cam_pitch, cam_roll] = quat2angle(cam_quat,'ZYX');
[target_yaw, target_pitch, target_roll] = quat2angle(target_quat,'ZYX');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (2) Use the "Initial Camera" approach:
%     We treat the camera's first frame as if it is at (0,0,0) with no rotation.
%     This means all subsequent target positions get transformed
%     into that frame.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2a) Grab the initial camera pose
cam_pos_0  = cam_pos(1,:);      % The camera's world position at frame 1
cam_quat_0 = cam_quat(1,:);     % The camera's world orientation at frame 1

% 2b) Convert that quaternion to a rotation matrix.
Rwc0 = quat2rotm(cam_quat_0);

% 2c) We want "world -> camera(0)" transform, so we define:
Rcw0 = Rwc0';   % Transpose to invert the rotation if necessary.

% 2d) Transform the target from the world frame into the camera(0) frame
T_in_cam0 = zeros(data_len, 3);  % store all frames
rot_in_cam0 = zeros(data_len, 3); % store rotations in camera frame
for i = 1:data_len
    % Vector from the *initial camera(0)* to the target(i) in world coords:
    vec_w = target_pos(i,:) - cam_pos_0;  % [T_w(i) - C_w(0)]
    
    % Now rotate that vector into camera(0) coords
    T_in_cam0(i,:) = (Rcw0 * vec_w')';
    
    % Convert target quaternion relative to the initial camera quaternion
    rel_quat = quatmultiply(quatinv(cam_quat_0), target_quat(i,:));
    [rel_yaw, rel_pitch, rel_roll] = quat2angle(rel_quat, 'ZYX');
    rot_in_cam0(i,:) = [rel_roll, rel_pitch, rel_yaw] * rad2deg;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (3) Plot the target in the "initial camera" frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure; 
hold on; axis equal; grid on;
title('Target in the Frame of the INITIAL Camera Pose');
xlabel('X_{cam0} [m]');
ylabel('Y_{cam0} [m]');
zlabel('Z_{cam0} [m]');

% Plot the camera at frame 0 as (0,0,0)
plot3(0,0,0,'ro','MarkerSize',10,'MarkerFaceColor','r');
text(0,0,0,' Camera(0)','Color','r');

% Draw small reference axes for the camera
scale_cam_axes = 0.1;
quiver3(0,0,0, scale_cam_axes,0,0, 'r','LineWidth',2);
text(scale_cam_axes,0,0,'X_{cam0}','Color','r');
quiver3(0,0,0, 0,scale_cam_axes,0, 'g','LineWidth',2);
text(0,scale_cam_axes,0,'Y_{cam0}','Color','g');
quiver3(0,0,0, 0,0,scale_cam_axes, 'b','LineWidth',2);
text(0,0,scale_cam_axes,'Z_{cam0}','Color','b');

% Plot the transformed target points
plot3(T_in_cam0(:,1), T_in_cam0(:,2), T_in_cam0(:,3), ...
      'b.-','MarkerSize',15);

legend('Camera(0) origin','Camera(0) axes','Target (in cam(0) frame)',...
       'Location','best');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (4) Plot the relative rotation (roll, pitch, yaw) vs time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
hold on; grid on;
title('Relative Rotation of Target in Initial Camera Frame');
xlabel('Frame Index');
ylabel('Rotation [deg]');

plot(rot_in_cam0(:,1), 'r-', 'LineWidth', 2); % Roll
plot(rot_in_cam0(:,2), 'g-', 'LineWidth', 2); % Pitch
plot(rot_in_cam0(:,3), 'b-', 'LineWidth', 2); % Yaw

legend('Roll','Pitch','Yaw','Location','best');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (5) Plot the relative positions (x, y, z) vs time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
hold on; grid on;
title('Relative Position of Target in Initial Camera Frame');
xlabel('Frame Index');
ylabel('Position [m]');

plot(T_in_cam0(:,1), 'r-', 'LineWidth', 2); % X
plot(T_in_cam0(:,2), 'g-', 'LineWidth', 2); % Y
plot(T_in_cam0(:,3), 'b-', 'LineWidth', 2); % Z

legend('X','Y','Z','Location','best');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (6) Calculate and display statistics for relative positions and rotations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Relative position statistics
avg_pos = mean(T_in_cam0, 1);
min_pos = min(T_in_cam0, [], 1);
max_pos = max(T_in_cam0, [], 1);

% Relative rotation statistics
avg_rot = mean(rot_in_cam0, 1);
min_rot = min(rot_in_cam0, [], 1);
max_rot = max(rot_in_cam0, [], 1);

% Display statistics
fprintf('Relative Position Statistics (X, Y, Z):\n');
fprintf('  Average: [%.2f, %.2f, %.2f] m\n', avg_pos);
fprintf('  Minimum: [%.2f, %.2f, %.2f] m\n', min_pos);
fprintf('  Maximum: [%.2f, %.2f, %.2f] m\n', max_pos);

fprintf('\nRelative Rotation Statistics (Roll, Pitch, Yaw):\n');
fprintf('  Average: [%.2f, %.2f, %.2f] deg\n', avg_rot);
fprintf('  Minimum: [%.2f, %.2f, %.2f] deg\n', min_rot);
fprintf('  Maximum: [%.2f, %.2f, %.2f] deg\n', max_rot);
