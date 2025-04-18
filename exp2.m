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

    %% 2. Median Filtering (per channel)
    med = zeros(size(img));
    for c = 1:3
        med(:,:,c) = medfilt2(img(:,:,c), [3 3]); % Example: 3x3 median window
    end

    %% 3. Unsharp Masking (per channel)
    unsharp = zeros(size(img));
    for c = 1:3
        unsharp(:,:,c) = imsharpen(img(:,:,c), ...
            'Radius', 1, ...
            'Amount', 1, ...
            'Threshold', 0);
    end

    %% Evaluate Metrics
    methods = {'gaussian', 'median', 'unsharp'};
    images = {img_gaussian, med, unsharp};

    for j = 1:length(methods)
        method = methods{j};
        img_method = images{j};

        % PSNR (Peak Signal-to-Noise Ratio)
        psnr_val = psnr(rgb2gray(img_method), img_gray);

        % MSE (Mean Squared Error)
        mse_val = immse(rgb2gray(img_method), img_gray);

        % Edge Strength Analysis (Sobel Operator & Laplacian Variance)
        sobel_val = sum(sum(edge(rgb2gray(img_method), 'sobel')));
        laplacian_val = var(double(del2(rgb2gray(img_method))));

        % Ring Artifact Detection (for Unsharp Masking)
        if strcmp(method, 'unsharp')
            ring_artifact_val = sum(sum(edge(rgb2gray(img_method), 'log')));
        else
            ring_artifact_val = NaN; % Not applicable for other methods
        end

        % ========================= Save Processed Images (Color) =========================
        imwrite(img_gaussian, fullfile(output_folder, ['gaussian_' img_name]));
        imwrite(med, fullfile(output_folder, ['median_' img_name]));
        imwrite(unsharp, fullfile(output_folder, ['unsharp_' img_name]));

        % ========================= Store Results =========================
        results = [results; {img_name, ...
            psnr_val, mse_val, sobel_val, laplacian_val, ring_artifact_val}];
    end % End of methods loop
end % End of image processing loop

% Convert results to table and save
results_table = cell2table(results, ...
    'VariableNames', {'Image', 'PSNR', 'MSE', 'Sobel', 'Laplacian', 'Ring_Artifact'});

% Save results to CSV file
writetable(results_table, fullfile(output_folder, 'exp2_results.csv'));

disp('Experiment 2 completed. Results saved.');
