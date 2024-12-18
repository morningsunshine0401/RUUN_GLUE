clc; clear; close all;
folderPath = "test1_box.db3";
bagReader = ros2bagreader(folderPath);
% baginfo = ros2("bag","info",folderPath);

msgs = readMessages(bagReader);
msgs_mat_box = cell2mat(msgs);

