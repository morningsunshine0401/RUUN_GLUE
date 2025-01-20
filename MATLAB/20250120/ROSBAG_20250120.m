%% Clear and setup
clear all; close all; clc;

bagFile = "20250120_test5.db3";  % Replace with your .db3 path
bagReader  = ros2bagreader(bagFile);
messageList = bagReader.MessageList;
allMsgs     = readMessages(bagReader);

%% Identify rows for each topic
imageIdx = find(messageList.Topic == "/webcam_image");
optiIdx  = find(messageList.Topic == "/OPTI/rb_infos");

%% Extract camera data
imageTimes = zeros(numel(imageIdx),1);
imageData  = cell(numel(imageIdx),1);

for i = 1:numel(imageIdx)
    row = imageIdx(i);

    % Convert time to numeric
    if isdatetime(messageList.Time(row))
        imageTimes(i) = posixtime(messageList.Time(row));
    else
        imageTimes(i) = messageList.Time(row);
    end
    
    % sensor_msgs/Image -> MATLAB image
    imageData{i} = rosReadImage(allMsgs{row});
end

%% Extract OptiTrack data
optiTimes = zeros(numel(optiIdx),1);
optiData  = cell(numel(optiIdx),1);

for i = 1:numel(optiIdx)
    row = optiIdx(i);

    if isdatetime(messageList.Time(row))
        optiTimes(i) = posixtime(messageList.Time(row));
    else
        optiTimes(i) = messageList.Time(row);
    end
    
    % Float32MultiArray data is likely "allMsgs{row}.data"
    optiData{i} = allMsgs{row}.data;
end

%% Offline nearest-time matching and display
figure('Name','Offline Sync','NumberTitle','off');

for i = 1:numel(imageIdx)
    thisImgTime = imageTimes(i);
    
    % Find the closest OptiTrack message in time
    [~, idxClosest] = min(abs(optiTimes - thisImgTime));
    
    subplot(1,2,1);
    imshow(imageData{i});
    title(sprintf("Webcam (t=%.3f)", thisImgTime));
    
    subplot(1,2,2);
    plot(optiData{idxClosest}, 'o-');
    title(sprintf("OPTI (t=%.3f)", optiTimes(idxClosest)));
    xlabel('Index in Float32 array');
    ylabel('Value');
    
    drawnow;
    pause(0.5);
end