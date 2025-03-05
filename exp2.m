% Experiment 2: Edge-Preserving Noise Reduction and Sharpness Enhancement
% MATLAB script to compare Gaussian Smoothing, Median Filtering, and Unsharp Masking with color output

clc; clear; close all;

% Define the dataset folder
input_folder = 'dataset/exp2/';
output_folder = 'exp2_results/';

% Create output directory if it doesn't exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Get list of images in the dataset folder
image_files = dir(fullfile(input_folder, '*.jpg')); % Change to '*.png' if needed
num_images = length(image_files);

% Preallocate results table
results = [];

% Loop through all images in the dataset
for i = 1:num_images
    % Read image
    img_name = image_files(i).name;
    img_path = fullfile(input_folder, img_name);
    img = im2double(imread(img_path)); % Convert to double for processing

    % Convert to grayscale for evaluation metrics
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end

    % ========================= Filtering Techniques (Color Processing) =========================
    
    % Gaussian Smoothing (Applied to Each RGB Channel)
    img_gaussian = zeros(size(img));
    for c = 1:size(img, 3)
        img_gaussian(:,:,c) = imgaussfilt(img(:,:,c), 1); % Apply Gaussian smoothing to each channel
    end

    % Median Filtering (Applied to Each RGB Channel)
    img_median = zeros(size(img));
    for c = 1:size(img, 3)
        img_median(:,:,c) = medfilt2(img(:,:,c), [5 5]); % Apply median filtering to each channel
    end

    % Unsharp Masking (Applied to Each RGB Channel)
    img_unsharp = zeros(size(img));
    for c = 1:size(img, 3)
        blurred = imgaussfilt(img(:,:,c), 2); % Blur each channel
        img_unsharp(:,:,c) = img(:,:,c) + (img(:,:,c) - blurred); % Apply sharpening
    end

    % ========================= Evaluation Metrics (Grayscale) =========================
    
    % SSIM (Structural Similarity Index)
    ssim_gaussian = ssim(rgb2gray(img_gaussian), img_gray);
    ssim_median = ssim(rgb2gray(img_median), img_gray);
    ssim_unsharp = ssim(rgb2gray(img_unsharp), img_gray);

    % PSNR (Peak Signal-to-Noise Ratio)
    psnr_gaussian = psnr(rgb2gray(img_gaussian), img_gray);
    psnr_median = psnr(rgb2gray(img_median), img_gray);
    psnr_unsharp = psnr(rgb2gray(img_unsharp), img_gray);

    % MSE (Mean Squared Error)
    mse_gaussian = immse(rgb2gray(img_gaussian), img_gray);
    mse_median = immse(rgb2gray(img_median), img_gray);
    mse_unsharp = immse(rgb2gray(img_unsharp), img_gray);

    % Edge Strength Analysis (Sobel Operator & Laplacian Variance)
    sobel_original = sum(sum(edge(img_gray, 'sobel')));
    sobel_gaussian = sum(sum(edge(rgb2gray(img_gaussian), 'sobel')));
    sobel_median = sum(sum(edge(rgb2gray(img_median), 'sobel')));
    sobel_unsharp = sum(sum(edge(rgb2gray(img_unsharp), 'sobel')));

    laplacian_original = var(double(del2(img_gray)));
    laplacian_gaussian = var(double(del2(rgb2gray(img_gaussian))));
    laplacian_median = var(double(del2(rgb2gray(img_median))));
    laplacian_unsharp = var(double(del2(rgb2gray(img_unsharp))));

    % Ring Artifact Detection (Detects unwanted halos from over-sharpening)
    ring_artifact_unsharp = sum(sum(edge(rgb2gray(img_unsharp), 'log')));

    % ========================= Save Processed Images (Color) =========================
    
    imwrite(img_gaussian, fullfile(output_folder, ['gaussian_' img_name]));
    imwrite(img_median, fullfile(output_folder, ['median_' img_name]));
    imwrite(img_unsharp, fullfile(output_folder, ['unsharp_' img_name]));

    % ========================= Store Results =========================
    
    results = [results; {img_name, ...
        ssim_gaussian, psnr_gaussian, mse_gaussian, sobel_gaussian, laplacian_gaussian, ...
        ssim_median, psnr_median, mse_median, sobel_median, laplacian_median, ...
        ssim_unsharp, psnr_unsharp, mse_unsharp, sobel_unsharp, laplacian_unsharp, ring_artifact_unsharp}];
end

% Convert results to table and save
results_table = cell2table(results, ...
    'VariableNames', {'Image', ...
    'SSIM_Gaussian', 'PSNR_Gaussian', 'MSE_Gaussian', 'Sobel_Gaussian', 'Laplacian_Gaussian', ...
    'SSIM_Median', 'PSNR_Median', 'MSE_Median', 'Sobel_Median', 'Laplacian_Median', ...
    'SSIM_Unsharp', 'PSNR_Unsharp', 'MSE_Unsharp', 'Sobel_Unsharp', 'Laplacian_Unsharp', 'Ring_Artifact_Unsharp'});

% Save results to CSV file
writetable(results_table, fullfile(output_folder, 'exp2_results.csv'));

disp('Experiment 2 completed. Results saved.');
