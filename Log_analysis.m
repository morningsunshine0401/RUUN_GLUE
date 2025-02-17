%clc; clear; close all;

% Path to log file
log_file_path = "pose_estimator.log";

% Open the log file and read lines
fid = fopen(log_file_path, 'r');
lines = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
lines = lines{1};

% Initialize variables
frames = [];
inliers = [];
inlier_ratio = [];
mean_reproj_error = [];
viewpoint_diff = [];
mahalanobis_sq = [];
coverage_score = [];
translation_change = [];
orientation_change = [];
alpha_values = [];
max_translation_jump = [];
max_orientation_jump = [];
correction_type = strings(0);
full_correction_reason = strings(0);

% Regular expressions for extracting data
frame_pattern = "Frame (\d+):";
inlier_pattern = "inliers=(\d+)";
inlier_ratio_pattern = "inlier_ratio=([\d.]+)";
mean_reproj_pattern = "mean_reprojection_error=([\d.]+)";
viewpoint_diff_pattern = "viewpoint_diff=([\d.]+)°";
mahalanobis_sq_pattern = "mahalanobis_sq=([\d.]+)";
coverage_score_pattern = "coverage_score=([\d.]+)";
translation_change_pattern = "translation_change=([\d.]+)";
orientation_change_pattern = "orientation_change_deg=([\d.]+)°";
alpha_pattern = "Partial blend applied with alpha=([\d.]+)";
max_translation_pattern = "max_translation_jump_adapt=([\d.]+)";
max_orientation_pattern = "max_orientation_jump_adapt_deg=([\d.]+)";

% Loop through log file lines and extract data
for i = 1:length(lines)
    line = lines{i};

    frame_match = regexp(line, frame_pattern, 'tokens');
    if ~isempty(frame_match)
        frames(end+1, 1) = str2double(frame_match{1}{1});
        
        inliers(end+1, 1) = extract_numeric(line, inlier_pattern);
        inlier_ratio(end+1, 1) = extract_numeric(line, inlier_ratio_pattern);
        mean_reproj_error(end+1, 1) = extract_numeric(line, mean_reproj_pattern);
        viewpoint_diff(end+1, 1) = extract_numeric(line, viewpoint_diff_pattern);
        mahalanobis_sq(end+1, 1) = extract_numeric(line, mahalanobis_sq_pattern);
        coverage_score(end+1, 1) = extract_numeric(line, coverage_score_pattern);
        translation_change(end+1, 1) = extract_numeric(line, translation_change_pattern);
        orientation_change(end+1, 1) = extract_numeric(line, orientation_change_pattern);
        alpha_values(end+1, 1) = extract_numeric(line, alpha_pattern);
        max_translation_jump(end+1, 1) = extract_numeric(line, max_translation_pattern);
        max_orientation_jump(end+1, 1) = extract_numeric(line, max_orientation_pattern);

        % Determine correction type
        if contains(line, "Kalman Filter correction accepted due to")
            correction_type(end+1, 1) = "Full Correction";
            reason_match = regexp(line, "accepted due to (.+?)\.", 'tokens');
            if ~isempty(reason_match)
                full_correction_reason(end+1, 1) = reason_match{1}{1};
            else
                full_correction_reason(end+1, 1) = "Unknown";
            end
        elseif contains(line, "Partial blend applied with alpha")
            correction_type(end+1, 1) = "Partial Blend";
            full_correction_reason(end+1, 1) = "Partial Blend";
        elseif contains(line, "Exceeded max skip count")
            correction_type(end+1, 1) = "Forced Correction";
            full_correction_reason(end+1, 1) = "Forced Correction";
        else
            correction_type(end+1, 1) = "";
            full_correction_reason(end+1, 1) = "";
        end
    end
end

% Convert extracted data to table
T = table(frames, inliers, inlier_ratio, mean_reproj_error, viewpoint_diff, ...
    mahalanobis_sq, coverage_score, translation_change, orientation_change, ...
    alpha_values, max_translation_jump, max_orientation_jump, correction_type, full_correction_reason, ...
    'VariableNames', {'Frame', 'Inliers', 'InlierRatio', 'MeanReprojError', 'ViewpointDiff', ...
    'MahalanobisSq', 'CoverageScore', 'TranslationChange', 'OrientationChange', ...
    'Alpha', 'MaxTranslationJump', 'MaxOrientationJump', 'CorrectionType', 'FullCorrectionReason'});

% Save to CSV and JSON
writetable(T, 'extracted_pose_data.csv');
jsonStr = jsonencode(T);
fid = fopen('extracted_pose_data.json', 'w');
fprintf(fid, '%s', jsonStr);
fclose(fid);

% ---------- Plot Data ----------

figure;
tiledlayout(3,2);

% Inliers vs Frame
nexttile;
plot(T.Frame, T.Inliers, 'bo-');
xlabel('Frame'); ylabel('Inliers'); title('Inliers vs Frame');

% Coverage Score vs Frame
nexttile;
plot(T.Frame, T.CoverageScore, 'go-');
xlabel('Frame'); ylabel('Coverage Score'); title('Coverage Score vs Frame');

% Mahalanobis Sq vs Frame
nexttile;
plot(T.Frame, T.MahalanobisSq, 'ro-');
xlabel('Frame'); ylabel('Mahalanobis Sq'); title('Mahalanobis Sq vs Frame');

% Translation Change vs Frame
nexttile;
plot(T.Frame, T.TranslationChange, 'mo-');
xlabel('Frame'); ylabel('Translation Change'); title('Translation Change vs Frame');

% Max Translation Jump vs Frame
nexttile;
plot(T.Frame, T.MaxTranslationJump, 'ro--');
xlabel('Frame'); ylabel('Max Translation Jump');
title('Max Translation Jump vs Frame');
ylim([0 prctile(T.MaxTranslationJump, 95)]); % Cap extreme values

% Orientation Change vs Frame
nexttile;
plot(T.Frame, T.OrientationChange, 'co-');
xlabel('Frame'); ylabel('Orientation Change (°)');
title('Orientation Change vs Frame');

% Max Orientation Jump vs Frame
figure;
plot(T.Frame, T.MaxOrientationJump, 'r--');
xlabel('Frame'); ylabel('Max Orientation Jump');
title('Max Orientation Jump vs Frame');
ylim([0 prctile(T.MaxOrientationJump, 95)]);

% Full Correction Frames Visualization
figure;
full_correction_idx = strcmp(T.CorrectionType, "Full Correction");
if any(full_correction_idx)
    categories = unique(T.FullCorrectionReason(full_correction_idx));
    hold on;
    for i = 1:length(categories)
        category_idx = strcmp(T.FullCorrectionReason, categories{i});
        scatter(T.Frame(category_idx), ones(sum(category_idx), 1) * i, 100, 'filled', 'DisplayName', categories{i});
    end
    hold off;
    xlabel('Frame'); title('Frames with Full Correction and Their Reasons');
    legend;
end

% Function to extract numeric values from a line using regex
function num = extract_numeric(line, pattern)
    match = regexp(line, pattern, 'tokens');
    if isempty(match)
        num = NaN;
    else
        num = str2double(match{1}{1});
    end
end
