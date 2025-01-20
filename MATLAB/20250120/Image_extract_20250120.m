%% (0) Preparations
clear all; close all; clc;

%% Specify ROS 2 bag file and output directory
bagFile      = "20250120_test5.db3";   % <-- Change to your actual .db3
outputFolder = "extracted_images_test5";

%% 1. Read the bag
bagReader   = ros2bagreader(bagFile);
messageList = bagReader.MessageList;
allMsgs     = readMessages(bagReader);

%% 2. Find indices for /webcam_image
imgIdx = find(messageList.Topic == "/webcam_image");
N_img  = numel(imgIdx);
disp("Number of images found: " + N_img);

%% 3. Create an output folder if not existing
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

%% 4. Prepare a table to store (Index, Timestamp, Filename)
imageTable = table('Size',[N_img 3], ...
                   'VariableTypes',{'double','double','string'}, ...
                   'VariableNames',{'Index','Timestamp','Filename'});

%% 5. Extract each image, save to disk, record info in table
for i = 1:N_img
    row = imgIdx(i);

    % A) Extract the timestamp
    if isdatetime(messageList.Time(row))
        tSec = posixtime(messageList.Time(row));  % convert datetime -> numeric
    else
        tSec = messageList.Time(row);  % already numeric in some MATLAB versions
    end

    % B) Convert sensor_msgs/Image => MATLAB image
    imgMat = rosReadImage(allMsgs{row});

    % C) Create a filename with frame index & timestamp
    %    e.g., "frame_00001_1234567890.123.png"
    frameName = sprintf("frame_%05d_%.3f.png", i, tSec);
    outPath   = fullfile(outputFolder, frameName);

    % D) Write the image to disk
    imwrite(imgMat, outPath);

    % E) Store the metadata in our table
    imageTable.Index(i)     = i;
    imageTable.Timestamp(i) = tSec;
    imageTable.Filename(i)  = frameName;
end

%% 6. Save a CSV and a MAT file with the table
csvFile = fullfile(outputFolder, "image_index.csv");
matFile = fullfile(outputFolder, "image_index.mat");

writetable(imageTable, csvFile);
save(matFile, "imageTable");

disp("Finished extracting images. Files saved in: " + outputFolder);
disp("Image index CSV: " + csvFile);