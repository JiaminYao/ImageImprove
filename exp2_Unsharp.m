clc; clear; close all;

%% Parameters
imageDir = 'dataset/exp2';
outputDir = 'exp2_Unsharp';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

imageFiles = dir(fullfile(imageDir, '*.jpg'));

% Only optimize unsharpAmount
unsharpAmount_values = 0.5:0.5:2.0;

% Fixed parameters
unsharpRadius = 1;
unsharpThreshold = 0;

% Metric weights (same as Gaussian)
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

    score_list = zeros(size(unsharpAmount_values));
    metric_table = [];

    for s = 1:length(unsharpAmount_values)
        amount = unsharpAmount_values(s);
        sharpened = zeros(size(orig));
        for c = 1:3
            sharpened(:,:,c) = imsharpen(orig(:,:,c), ...
                'Radius', unsharpRadius, ...
                'Amount', amount, ...
                'Threshold', unsharpThreshold);
        end

        % Compute metrics
        ssim_val = compute_ssim(orig, sharpened);
        psnr_val = compute_psnr(orig, sharpened);
        if isinf(psnr_val); psnr_val = 100; end
        mse_val  = compute_mse(orig, sharpened);
        edge     = compute_edge_strength(sharpened);
        ring     = ring_artifact_detection(sharpened);

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
            amount, ssim_val, psnr_val, mse_val, edge.sobel, edge.laplacianVar, ring, score
        }];
    end

    %% Convert to Table and Save
    T = cell2table(metric_table, ...
        'VariableNames', {'Amount', 'SSIM', 'PSNR', 'MSE', 'Sobel', 'LaplacianVar', 'Ring', 'Score'});
    writetable(T, fullfile(outputDir, ['metrics_' erase(imgName, '.jpg') '.csv']));

    %% Find best amount and save image
    [~, bestIdx] = max(score_list);
    bestAmount = unsharpAmount_values(bestIdx);
    bestImage = zeros(size(orig));
    for c = 1:3
        bestImage(:,:,c) = imsharpen(orig(:,:,c), ...
            'Radius', unsharpRadius, ...
            'Amount', bestAmount, ...
            'Threshold', unsharpThreshold);
    end
    imwrite(bestImage, fullfile(outputDir, ['best_' imgName]));

    %% Show Result
    figure('Name', ['Best Unsharp Amount - ' imgName]);
    subplot(1,2,1); imshow(orig); title('Original');
    subplot(1,2,2); imshow(bestImage); title(sprintf('Best Unsharp Amount = %.1f', bestAmount));
end

fprintf('✅ All results saved to folder: %s\n', outputDir);
