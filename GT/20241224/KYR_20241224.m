clear all;
close all;
clc;

figure_on = 1;

rad2deg = 180/pi;

folderPath = 'test20241224\test3.db3';

bagReader = ros2bagreader(folderPath);
baginfo = ros2("bag","info",folderPath);
msgs = readMessages(bagReader);

OPTI_rb_infos           = select(bagReader,"Topic","/OPTI/rb_infos");

OPTI_rb_infos_Filtered   = readMessages(OPTI_rb_infos);

data_len = length(OPTI_rb_infos_Filtered);
for i=1:data_len
    OPTI_rb_infos_data(i,:)   = cell2mat(OPTI_rb_infos_Filtered(i,1)).data';
end
OPTI_rb_infos_time = (      OPTI_rb_infos.MessageList.Time - ...
                                min(OPTI_rb_infos.MessageList.Time));


cam_pos = OPTI_rb_infos_data(:,1:3);
cam_quat = OPTI_rb_infos_data(:,4:7);
[cam_yaw, cam_pitch, cam_roll] = quat2angle(cam_quat,'ZYX');

target_pos = OPTI_rb_infos_data(:,8:10);
target_quat = OPTI_rb_infos_data(:,11:14);
[target_yaw, target_pitch, target_roll] = quat2angle(target_quat,'ZYX');


if(figure_on)

figure;
subplot(3,1,1);
plot(OPTI_rb_infos_time, cam_yaw*rad2deg);
title('CAM Attitude');
grid on;
ylabel('yaw [deg]');
xlabel('Time[sec]');
subplot(3,1,2);
plot(OPTI_rb_infos_time, cam_pitch*rad2deg);
grid on;
ylabel('pitch [deg]');
xlabel('Time[sec]');
subplot(3,1,3);
plot(OPTI_rb_infos_time, cam_roll*rad2deg);
grid on;
ylabel('roll [deg]');
xlabel('Time[sec]');

figure;
subplot(3,1,1);
plot(OPTI_rb_infos_time, target_yaw*rad2deg);
title('TARGET Attitude');
grid on;
ylabel('yaw [deg]');
xlabel('Time[sec]');
subplot(3,1,2);
plot(OPTI_rb_infos_time, target_pitch*rad2deg);
grid on;
ylabel('pitch [deg]');
xlabel('Time[sec]');
subplot(3,1,3);
plot(OPTI_rb_infos_time, target_roll*rad2deg);
grid on;
ylabel('roll [deg]');
xlabel('Time[sec]');

figure;
subplot(3,1,1);
plot(OPTI_rb_infos_time, cam_pos(:,1)*100);
title('CAM Position')
grid on;
ylabel('X [cm]');
xlabel('Time[sec]');
subplot(3,1,2);
plot(OPTI_rb_infos_time, cam_pos(:,2)*100);
grid on;
ylabel('Y [cm]');
xlabel('Time[sec]');
subplot(3,1,3);
plot(OPTI_rb_infos_time, cam_pos(:,3)*100);
grid on;
ylabel('Z [cm]');
xlabel('Time[sec]');

figure;
subplot(3,1,1);
plot(OPTI_rb_infos_time, target_pos(:,1)*100);
title('TARGET Position')
grid on;
ylabel('X [cm]');
xlabel('Time[sec]');
subplot(3,1,2);
plot(OPTI_rb_infos_time, target_pos(:,2)*100);
grid on;
ylabel('Y [cm]');
xlabel('Time[sec]');
subplot(3,1,3);
plot(OPTI_rb_infos_time, target_pos(:,3)*100);
grid on;
ylabel('Z [cm]');
xlabel('Time[sec]');

end

