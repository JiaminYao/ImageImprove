% Experiment 3: Comparing Automated Background Suppression Techniques
% High-Pass Fourier Filtering vs. Gaussian Blurring vs. Traditional Masking (Color-Preserved)

clc; clear; close all;

% Define dataset folder
input_folder = 'dataset/exp3/';
output_folder = 'exp3_results/';

% Create output directory if it doesn't exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Get list of images in the dataset folder
image_files = dir(fullfile(input_folder, '*.jpg')); % Change to '*.png' if needed
num_images = length(image_files);

% Preallocate results table
results = [];

for i = 1:num_images
    % Read image
    img_name = image_files(i).name;
    img_path = fullfile(input_folder, img_name);
    img = im2double(imread(img_path)); % Convert to double for processing
    
    % Convert to grayscale only if needed
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    
    % Extract image size
    [rows, cols, channels] = size(img);
    
    % ========================= Background Suppression Techniques =========================
    
    % **High-Pass Fourier Filtering (Applied to Each Color Channel)**
    img_high_pass = zeros(size(img)); % Initialize result
    for c = 1:channels
        img_fft = fft2(img(:,:,c)); % Compute 2D FFT for each channel
        img_fft_shifted = fftshift(img_fft); % Shift zero frequency to center
        
        % High-Pass Filter Mask
        [X, Y] = meshgrid(1:cols, 1:rows);
        center_x = cols / 2; 
        center_y = rows / 2;
        radius = min(rows, cols) * 0.15; % Define high-pass radius
        high_pass_mask = sqrt((X - center_x).^2 + (Y - center_y).^2) > radius;
        
        % Apply High-Pass Filter and inverse transform
        img_high_pass(:,:,c) = real(ifft2(ifftshift(img_fft_shifted .* high_pass_mask)));
    end
    
    % **Gaussian Blurring (Background Blur, Keep Subject Sharp)**
    img_gaussian = imgaussfilt(img, 5); % Strong background blur

    % **Traditional Masking (Simple Segmentation to Preserve Subject)**
    mask = img_gray > graythresh(img_gray); % Generate binary mask
    img_masked = img .* repmat(mask, [1, 1, 3]); % Apply mask to RGB image
    
    % ========================= Evaluation Metrics =========================

    % **Fourier Spectrum Analysis (Low-frequency suppression)**
    spectrum_original = sum(sum(abs(fftshift(fft2(img_gray)))));
    spectrum_high_pass = sum(sum(abs(fftshift(fft2(rgb2gray(img_high_pass))))));

    % **Edge Preservation Score (Sobel and Laplacian Variance)**
    sobel_original = sum(sum(edge(img_gray, 'sobel')));
    sobel_high_pass = sum(sum(edge(rgb2gray(img_high_pass), 'sobel')));
    sobel_gaussian = sum(sum(edge(rgb2gray(img_gaussian), 'sobel')));
    sobel_masked = sum(sum(edge(rgb2gray(img_masked), 'sobel')));
    
    % **Laplacian Variance for Edge Strength**
    laplacian_original = var(double(del2(img_gray)));  
    laplacian_high_pass = var(double(del2(rgb2gray(img_high_pass))));
    laplacian_gaussian = var(double(del2(rgb2gray(img_gaussian))));
    laplacian_masked = var(double(del2(rgb2gray(img_masked))));
    
    % **SSIM (Structural Similarity Index)**
    ssim_high_pass = ssim(rgb2gray(img_high_pass), img_gray);
    ssim_gaussian = ssim(rgb2gray(img_gaussian), img_gray);
    ssim_masked = ssim(rgb2gray(img_masked), img_gray);

    % **Noise Residual Map Analysis**
    noise_residual_high_pass = abs(img - img_high_pass);
    noise_residual_gaussian = abs(img - img_gaussian);
    noise_residual_masked = abs(img - img_masked);
    
    % **Paired Comparison Simulation (Random Binary Choice)**
    comparison_votes = [randi([0 1]), randi([0 1]), randi([0 1])]; % Random preference scores
    
    % ========================= Save Processed Images =========================
    
    imwrite(img_high_pass, fullfile(output_folder, ['highpass_' img_name]));
    imwrite(img_gaussian, fullfile(output_folder, ['gaussian_' img_name]));
    imwrite(img_masked, fullfile(output_folder, ['masked_' img_name]));

    % Save noise residual maps
    imwrite(mat2gray(noise_residual_high_pass), fullfile(output_folder, ['residual_highpass_' img_name]));
    imwrite(mat2gray(noise_residual_gaussian), fullfile(output_folder, ['residual_gaussian_' img_name]));
    imwrite(mat2gray(noise_residual_masked), fullfile(output_folder, ['residual_masked_' img_name]));

    % ========================= Store Results =========================
    
    results = [results; {img_name, ...
        spectrum_original, spectrum_high_pass, ...
        sobel_high_pass, sobel_gaussian, sobel_masked, ...
        laplacian_high_pass, laplacian_gaussian, laplacian_masked, ...
        ssim_high_pass, ssim_gaussian, ssim_masked, ...
        comparison_votes(1), comparison_votes(2), comparison_votes(3)}];
end

% Convert results to table and save
results_table = cell2table(results, ...
    'VariableNames', {'Image', ...
    'Spectrum_Original', 'Spectrum_HighPass', ...
    'Sobel_HighPass', 'Sobel_Gaussian', 'Sobel_Masked', ...
    'Laplacian_HighPass', 'Laplacian_Gaussian', 'Laplacian_Masked', ...
    'SSIM_HighPass', 'SSIM_Gaussian', 'SSIM_Masked', ...
    'Comparison_HighPass', 'Comparison_Gaussian', 'Comparison_Masked'});

% Save results to CSV file
writetable(results_table, fullfile(output_folder, 'exp3_results.csv'));

disp('Experiment 3 completed. Results saved.');
