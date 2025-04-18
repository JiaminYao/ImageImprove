clc; clear; close all;

%% Parameters
imageDir = 'dataset/exp2';
outputDir = 'exp2_Median';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

imageFiles = dir(fullfile(imageDir, '*.jpg'));

% Define window sizes to test
window_sizes = [3, 5, 7];

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

    score_list = zeros(size(window_sizes));
    metric_table = [];

    for s = 1:length(window_sizes)
        w = window_sizes(s);
        filtered = zeros(size(orig));
        for c = 1:3
            filtered(:,:,c) = medfilt2(orig(:,:,c), [w w], 'symmetric');
        end

        % Compute metrics
        ssim_val = compute_ssim(orig, filtered);
        psnr_val = compute_psnr(orig, filtered); if isinf(psnr_val); psnr_val = 100; end
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
            sprintf('%dx%d', w, w), ssim_val, psnr_val, mse_val, edge.sobel, edge.laplacianVar, ring, score
        }];
    end

    %% Convert to Table and Save
    T = cell2table(metric_table, ...
        'VariableNames', {'Window', 'SSIM', 'PSNR', 'MSE', 'Sobel', 'LaplacianVar', 'Ring', 'Score'});
    writetable(T, fullfile(outputDir, ['metrics_' erase(imgName, '.jpg') '.csv']));

    %% Find best window and save image
    [~, bestIdx] = max(score_list);
    bestWindow = window_sizes(bestIdx);
    bestImage = zeros(size(orig));
    for c = 1:3
        bestImage(:,:,c) = medfilt2(orig(:,:,c), [bestWindow bestWindow], 'symmetric');
    end
    imwrite(bestImage, fullfile(outputDir, ['best_' imgName]));

    %% Show Result
    figure('Name', ['Best Median Filter - ' imgName]);
    subplot(1,2,1); imshow(orig); title('Original');
    subplot(1,2,2); imshow(bestImage); title(sprintf('Best Window = %dx%d', bestWindow, bestWindow));
end

fprintf('✅ All results saved to folder: %s\n', outputDir);
