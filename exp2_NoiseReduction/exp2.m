clc; clear; close all;

%% Parameters
imageDir = 'dataset/exp2/';  % Directory containing color images
imageFiles = dir(fullfile(imageDir, '*.jpg'));
output_folder = 'exp2_results/';

gaussianSigma = 0.4;
unsharpAmount = 0.5;
unsharpRadius = 1;
unsharpThreshold = 0;
medianWindow = [3 3];

%% Create output folder if not exists
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

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
    orig = im2double(origRGB);  % RGB image in double

    res = struct();
    res.original = orig;

    %% 1. Gaussian Smoothing
    gauss = zeros(size(orig));
    for c = 1:3
        gauss(:,:,c) = imgaussfilt(orig(:,:,c), gaussianSigma);
    end

    %% 2. Median Filtering
    med = zeros(size(orig));
    for c = 1:3
        med(:,:,c) = medfilt2(orig(:,:,c), medianWindow);
    end

    %% 3. Unsharp Masking
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

        res.(method).image = img;
        res.(method).ssim = compute_ssim(orig, img);
        res.(method).psnr = compute_psnr(orig, img);
        res.(method).mse  = compute_mse(orig, img);
        res.(method).edges = compute_edge_strength(img);
        res.(method).ring = ring_artifact_detection(img);

        % Save processed image
        imwrite(img, fullfile(output_folder, [method '_' imgName]));
    end

    results.(sprintf('image_%d', k)) = res;
end

%% Save Metrics Summary to CSV
T = table();

for k = 1:length(imageFiles)
    imgID = sprintf('image_%d', k);
    imgName = imageFiles(k).name;

    for method = ["gaussian", "median", "unsharp"]
        r = results.(imgID).(method);
        newRow = table({imgName}, method, ...
            r.ssim, r.psnr, r.mse, ...
            r.edges.sobel, r.edges.laplacianVar, ...
            r.ring, ...
            'VariableNames', {'ImageName', 'Method', ...
            'SSIM', 'PSNR', 'MSE', ...
            'SobelEdgeStrength', 'LaplacianVariance', ...
            'RingArtifactScore'});

        T = [T; newRow];
    end
end

writetable(T, fullfile(output_folder, 'exp2_metrics_summary.csv'));

%% Optional: Visualization
for k = 1:length(imageFiles)
    imgID = sprintf('image_%d', k);
    figure('Name', ['Color Comparison - ' imageFiles(k).name]);

    subplot(2,2,1); imshow(results.(imgID).original); title('Original');
    subplot(2,2,2); imshow(results.(imgID).gaussian.image); title('Gaussian Smoothing');
    subplot(2,2,3); imshow(results.(imgID).median.image); title('Median Filtering');
    subplot(2,2,4); imshow(results.(imgID).unsharp.image); title('Unsharp Masking');
end
