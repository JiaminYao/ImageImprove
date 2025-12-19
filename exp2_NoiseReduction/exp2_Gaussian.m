clc; clear; close all;

%% Parameters
imageDir = 'dataset/exp2';
outputDir = 'exp2_Gaussian';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

imageFiles = dir(fullfile(imageDir, '*.jpg'));
sigma_values = 0.4:0.2:1;

% Metric weights
weights = struct(...
    'ssim', 0.4, ...
    'psnr', 0.3, ...
    'mse', -0.1, ...
    'sobel', 0.1, ...
    'laplacianVar', 0.1, ...
    'ring', -0.1 ...
);

%% Metric Functions
compute_ssim = @(a,b) ssim(a, b);
compute_psnr = @(a,b) psnr(a, b);
compute_mse  = @(a,b) mean((a(:) - b(:)).^2);
compute_edge_strength = @(img_rgb) struct( ...
    'sobel', mean(edge(rgb2gray(img_rgb), 'sobel'), 'all'), ...
    'laplacianVar', var(imfilter(rgb2gray(img_rgb), fspecial('laplacian', 0.2)), 0, 'all') ...
);
ring_artifact_detection = @(img_rgb) std(imfilter(rgb2gray(img_rgb), fspecial('laplacian'), 'replicate'), 0, 'all');

%% Process Each Image
for k = 1:length(imageFiles)
    imgName = imageFiles(k).name;
    imgPath = fullfile(imageDir, imgName);
    origRGB = imread(imgPath);
    orig = im2double(origRGB);
    
    score_list = zeros(size(sigma_values));
    metric_table = [];

    for s = 1:length(sigma_values)
        sigma = sigma_values(s);
        filtered = zeros(size(orig));
        for c = 1:3
            filtered(:,:,c) = imgaussfilt(orig(:,:,c), sigma);
        end

        % Compute metrics
        ssim_val = compute_ssim(orig, filtered);
        psnr_val = compute_psnr(orig, filtered);
        mse_val  = compute_mse(orig, filtered);
        edge     = compute_edge_strength(filtered);
        ring     = ring_artifact_detection(filtered);

        % Scoring formula
        score = ...
            weights.ssim * ssim_val + ...
            weights.psnr * psnr_val + ...
            weights.mse  * mse_val + ...
            weights.sobel * edge.sobel + ...
            weights.laplacianVar * edge.laplacianVar + ...
            weights.ring * ring;

        score_list(s) = score;

        % Store metrics
        metric_table = [metric_table; {
            sigma, ssim_val, psnr_val, mse_val, edge.sobel, edge.laplacianVar, ring, score
        }];
    end

    %% Convert to Table and Save
    T = cell2table(metric_table, ...
        'VariableNames', {'sigma', 'SSIM', 'PSNR', 'MSE', 'Sobel', 'LaplacianVar', 'Ring', 'Score'});
    writetable(T, fullfile(outputDir, ['metrics_' erase(imgName, '.jpg') '.csv']));

    %% Find best sigma and save image
    [~, bestIdx] = max(score_list);
    bestSigma = sigma_values(bestIdx);
    bestImage = zeros(size(orig));
    for c = 1:3
        bestImage(:,:,c) = imgaussfilt(orig(:,:,c), bestSigma);
    end
    imwrite(bestImage, fullfile(outputDir, ['best_' imgName]));

    %% Show Result
    figure('Name', ['Best Gaussian Sigma - ' imgName]);
    subplot(1,2,1); imshow(orig); title('Original');
    subplot(1,2,2); imshow(bestImage); title(sprintf('Best Gaussian (σ = %.1f)', bestSigma));
end

fprintf('✅ All results saved to folder: %s\n', outputDir);
