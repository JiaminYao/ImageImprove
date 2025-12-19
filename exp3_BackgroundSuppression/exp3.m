% exp3.m - Full script for Experiment 3
clc; clear; close all;

%% Setup input and output paths
input_folder = 'dataset/exp3';
output_folder = 'exp3_results';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

image_files = dir(fullfile(input_folder, '*.jpg'));
num_images = length(image_files);
results(num_images) = struct();

%% Process each image
for i = 1:num_images
    img_name = image_files(i).name;
    img_path = fullfile(input_folder, img_name);
    img = im2double(imread(img_path));

    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    [rows, cols, ~] = size(img);

    %% Generate Dog Mask
    T = graythresh(img_gray);
    mask = imbinarize(img_gray, T);
    if mean(img_gray(mask)) < mean(img_gray(~mask))
        mask = ~mask;
    end
    mask = imclose(mask, strel('disk', 5));
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 500);
    cc = bwconncomp(mask);
    stats = regionprops(cc, 'Area');
    [~, idx] = max([stats.Area]);
    dog_mask = false(size(mask));
    if ~isempty(idx)
        dog_mask(cc.PixelIdxList{idx}) = true;
    end

    %% Method 1: Luminance-Guided Background Blur
    img_luminance = img;
    lum = rgb2gray(img);
    inv_lum = 1 - mat2gray(lum);
    blur_weight = imgaussfilt(inv_lum, 10);
    blurred_strong = imgaussfilt(img, 10);
    for c = 1:3
        img_luminance(:,:,c) = img(:,:,c).*(1 - blur_weight) + blurred_strong(:,:,c).*blur_weight;
    end

    %% Method 2: Gaussian Blur on Background
    img_gaussian = img;
    blurred = imgaussfilt(img, 10);
    for c = 1:3
        img_gaussian(:,:,c) = img(:,:,c).*dog_mask + blurred(:,:,c).*(1 - dog_mask);
    end

    %% Method 3: Selective Gaussian Blur (Distance-based)
    distance_map = bwdist(dog_mask);
    sigma_map = mat2gray(distance_map);
    sigma_levels = [5, 15, 30, 50, 80];
    num_levels = length(sigma_levels);
    blurred_layers = cell(3, num_levels);
    for c = 1:3
        for s = 1:num_levels
            blurred_layers{c, s} = imgaussfilt(img(:,:,c), sigma_levels(s));
        end
    end
    sigma_idx = round(rescale(sigma_map, 1, num_levels));
    sigma_idx = min(max(sigma_idx, 1), num_levels);
    img_selective = img;
    for c = 1:3
        channel_selective = zeros(rows, cols);
        for s = 1:num_levels
            mask_level = (sigma_idx == s);
            blurred = blurred_layers{c, s};
            channel_selective(mask_level) = blurred(mask_level);
        end
        img_selective(:,:,c) = img(:,:,c).*dog_mask + channel_selective.*(1 - dog_mask);
    end

    %% Save results
    imwrite(img_luminance, fullfile(output_folder, ['luminance_' img_name]));
    imwrite(img_gaussian, fullfile(output_folder, ['gaussian_' img_name]));
    imwrite(img_selective, fullfile(output_folder, ['selective_' img_name]));
    imwrite(dog_mask, fullfile(output_folder, ['mask_' img_name]));

    %% Evaluation metrics
    E_orig = edge(img_gray, 'sobel');
    E_luminance = edge(rgb2gray(img_luminance), 'sobel');
    E_gauss = edge(rgb2gray(img_gaussian), 'sobel');
    E_selective = edge(rgb2gray(img_selective), 'sobel');
    overlap = @(A,B) sum(A(:) & B(:)) / sum(A(:));

    results(i).Image = img_name;
    results(i).Sobel_Luminance = overlap(E_orig, E_luminance);
    results(i).Sobel_Gaussian = overlap(E_orig, E_gauss);
    results(i).Sobel_Selective = overlap(E_orig, E_selective);

    results(i).SSIM_Luminance = ssim(rgb2gray(img_luminance), img_gray);
    results(i).SSIM_Gaussian = ssim(rgb2gray(img_gaussian), img_gray);
    results(i).SSIM_Selective = ssim(rgb2gray(img_selective), img_gray);

    img_gauss_gray = rgb2gray(img_gaussian);
    img_sel_gray = rgb2gray(img_selective);

    results(i).Entropy_Luminance = entropy(img_gray(~dog_mask));
    results(i).Entropy_Gaussian = entropy(img_gauss_gray(~dog_mask));
    results(i).Entropy_Selective = entropy(img_sel_gray(~dog_mask));

    results(i).FGEdge_Luminance = mean(edge(img_gray .* dog_mask, 'sobel'), 'all');
    results(i).FGEdge_Gaussian = mean(edge(img_gauss_gray .* dog_mask, 'sobel'), 'all');
    results(i).FGEdge_Selective = mean(edge(img_sel_gray .* dog_mask, 'sobel'), 'all');

    fg_std = std(img_gray(dog_mask));
    bg_std_lum = std(img_gray(~dog_mask));
    bg_std_gauss = std(img_gauss_gray(~dog_mask));
    bg_std_selective = std(img_sel_gray(~dog_mask));

    results(i).Contrast_Luminance = fg_std / (bg_std_lum + eps);
    results(i).Contrast_Gaussian = fg_std / (bg_std_gauss + eps);
    results(i).Contrast_Selective = fg_std / (bg_std_selective + eps);
end

%% Save final table
results_table = struct2table(results);
writetable(results_table, fullfile(output_folder, 'exp3_results.csv'));

%% Visualizations
methods = {'Luminance-Guided', 'Gaussian Blur', 'Selective Blur'};

% SSIM
figure;
bar([mean(results_table.SSIM_Luminance), mean(results_table.SSIM_Gaussian), mean(results_table.SSIM_Selective)]);
set(gca, 'XTickLabel', methods);
ylabel('Average SSIM');
title('Experiment 3: SSIM Comparison');
grid on;
saveas(gcf, fullfile(output_folder, 'exp3_ssim.png'));

% Sobel Edge Overlap
figure;
bar([mean(results_table.Sobel_Luminance), mean(results_table.Sobel_Gaussian), mean(results_table.Sobel_Selective)]);
set(gca, 'XTickLabel', methods);
ylabel('Sobel Edge Overlap');
title('Experiment 3: Edge Overlap');
grid on;
saveas(gcf, fullfile(output_folder, 'exp3_sobel.png'));

% Background Entropy
figure;
bar([mean(results_table.Entropy_Luminance), mean(results_table.Entropy_Gaussian), mean(results_table.Entropy_Selective)]);
set(gca, 'XTickLabel', methods);
ylabel('Background Entropy');
title('Experiment 3: Background Entropy');
grid on;
saveas(gcf, fullfile(output_folder, 'exp3_entropy.png'));

% Foreground Edge Strength
figure;
bar([mean(results_table.FGEdge_Luminance), mean(results_table.FGEdge_Gaussian), mean(results_table.FGEdge_Selective)]);
set(gca, 'XTickLabel', methods);
ylabel('Foreground Edge Strength');
title('Experiment 3: Foreground Edge Strength');
grid on;
saveas(gcf, fullfile(output_folder, 'exp3_fg_edge.png'));

% FG vs BG Contrast
figure;
bar([mean(results_table.Contrast_Luminance), mean(results_table.Contrast_Gaussian), mean(results_table.Contrast_Selective)]);
set(gca, 'XTickLabel', methods);
ylabel('FG/BG Contrast Ratio');
title('Experiment 3: Contrast Comparison');
grid on;
saveas(gcf, fullfile(output_folder, 'exp3_contrast.png'));

disp('✅ Experiment 3 complete. Outputs and charts saved.');
