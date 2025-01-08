clc; clear; close all;
folderPath = "test2.db3";
bagReader = ros2bagreader(folderPath);
% baginfo = ros2("bag","info",folderPath);

msgs = readMessages(bagReader);
msgs_mat_20250107_test2 = cell2mat(msgs);

