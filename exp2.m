clc; clear; close all;

outputDir = 'exp2_results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% Save images and compile results
results_table = [];

%% Parameters
imageDir = 'dataset/exp2';  % Directory containing color images
imageFiles = dir(fullfile(imageDir, '*.jpg'));

gaussianSigma = 0.2;

unsharpAmount = 0.5;
unsharpRadius = 1;
unsharpThreshold = 0;

medianWindow = [3 3];

%% Initialize Results
results = struct();

%% Helper Metric Functions
compute_ssim = @(a,b) ssim(a, b);
compute_psnr = @(a,b) psnr(a, b);
compute_mse  = @(a,b) mean((a(:) - b(:)).^2);

compute_edge_strength = @(img_rgb) struct( ...
    'sobel', mean(edge(rgb2gray(img_rgb), 'sobel'), 'all'), ...
    'laplacianVar', var(imfilter(rgb2gray(img_rgb), fspecial('laplacian', 0.2)), 0, 'all') ...
);

ring_artifact_detection = @(img_rgb) std(imfilter(rgb2gray(img_rgb), fspecial('laplacian'), 'replicate'), 0, 'all');

%% Loop through all images
for k = 1:length(imageFiles)
    imgName = imageFiles(k).name;
    imgPath = fullfile(imageDir, imgName);
    origRGB = imread(imgPath);
    orig = im2double(origRGB);  % Keep as RGB

    res = struct();
    res.original = orig;

    %% 1. Gaussian Smoothing (per channel)
    gauss = zeros(size(orig));
    for c = 1:3
        gauss(:,:,c) = imgaussfilt(orig(:,:,c), gaussianSigma);
    end

    %% 2. Median Filtering (per channel)
    med = zeros(size(orig));
    for c = 1:3
        med(:,:,c) = medfilt2(orig(:,:,c), medianWindow);
    %% 2. Median Filtering (per channel)
    med = zeros(size(orig));
    for c = 1:3
        med(:,:,c) = medfilt2(orig(:,:,c), medianWindow);
    end

    %% 3. Unsharp Masking (per channel)
    unsharp = zeros(size(orig));
    for c = 1:3
        unsharp(:,:,c) = imsharpen(orig(:,:,c), ...
            'Radius', unsharpRadius, ...
            'Amount', unsharpAmount, ...
            'Threshold', unsharpThreshold);
    %% 3. Unsharp Masking (per channel)
    unsharp = zeros(size(orig));
    for c = 1:3
        unsharp(:,:,c) = imsharpen(orig(:,:,c), ...
            'Radius', unsharpRadius, ...
            'Amount', unsharpAmount, ...
            'Threshold', unsharpThreshold);
    end

    %% Evaluate Metrics
    methods = {'gaussian', 'median', 'unsharp'};
    images = {gauss, med, unsharp};

    for i = 1:length(methods)
        method = methods{i};
        img = images{i};
    %% Evaluate Metrics
    methods = {'gaussian', 'median', 'unsharp'};
    images = {gauss, med, unsharp};

    for i = 1:length(methods)
        method = methods{i};
        img = images{i};

        res.(method).image = img;
        res.(method).ssim = compute_ssim(orig, img);
        res.(method).psnr = compute_psnr(orig, img);
        res.(method).mse  = compute_mse(orig, img);
        res.(method).edges = compute_edge_strength(img);
        res.(method).ring = ring_artifact_detection(img);
    end

    results.(sprintf('image_%d', k)) = res;
end

%% Print Summary
fprintf('\n--- Results Summary ---\n');
for k = 1:length(imageFiles)
    imgID = sprintf('image_%d', k);
    fprintf('\nImage %d: %s\n', k, imageFiles(k).name);

    for method = ["gaussian", "median", "unsharp"]
        r = results.(imgID).(method);
        fprintf('%s:\n', upper(method));
        fprintf('  SSIM: %.4f | PSNR: %.2f | MSE: %.4f\n', r.ssim, r.psnr, r.mse);
        fprintf('  Sobel Edge Strength: %.4f | Laplacian Variance: %.4f\n', ...
            r.edges.sobel, r.edges.laplacianVar);
        fprintf('  Ring Artifact Score: %.4f\n', r.ring);
    end
end

% Visualization for all images
for k = 1:length(imageFiles)
    imgID = sprintf('image_%d', k);
    figure('Name', ['Color Comparison - ' imageFiles(k).name]);

    subplot(2,2,1); imshow(results.(imgID).original); title('Original');
    subplot(2,2,2); imshow(results.(imgID).gaussian.image); title('Gaussian Smoothing');
    subplot(2,2,3); imshow(results.(imgID).median.image); title('Median Filtering');
    subplot(2,2,4); imshow(results.(imgID).unsharp.image); title('Unsharp Masking');
end

for k = 1:length(imageFiles)
    imgID = sprintf('image_%d', k);
    imgName = imageFiles(k).name;
    [~, baseName, ~] = fileparts(imgName);

    for method = ["gaussian", "median", "unsharp"]
        r = results.(imgID).(method);
        processed_img = r.image;

        % Save image
        output_img_path = fullfile(outputDir, sprintf('%s_%s.jpg', baseName, method));
        imwrite(processed_img, output_img_path);

        % Append metrics to table
        results_table = [results_table; {
            imgName, char(method), ...
            r.ssim, r.psnr, r.mse, ...
            r.edges.sobel, r.edges.laplacianVar, ...
            r.ring
        }];
    end
end

%% Save results as CSV
headers = {'Image_Name', 'Method', 'SSIM', 'PSNR', 'MSE', ...
           'Sobel_Edge', 'Laplacian_Var', 'Ring_Artifact_Score'};

results_table = cell2table(results_table, 'VariableNames', headers);
writetable(results_table, fullfile(outputDir, 'exp2_metrics_summary.csv'));

fprintf('\nAll processed images and metrics saved in: %s\n', outputDir);