% Experiment 1: Evaluating Localized Contrast Enhancement
% GHE, AHE, CLAHE

% Ideal Metric Trends for Best Results:
% Entropy: Should be moderately high (not excessive).
% Standard Deviation: Should increase (indicating better contrast).
% SSIM: Should remain high (preserves structure).
% Bhattacharyya Distance: Should be moderate (too low means little enhancement, too high means excessive change).
% LCV: Should increase but not excessively (ensuring local contrast is improved).

function exp1()
    % Load best parameters from an external file
    params_file = 'exp1_best_parameters.xlsx'; % Ensure the file is accessible
    params = readtable(params_file);

    % Define the dataset folder
    input_folder = 'dataset/exp1/';
    output_folder = 'exp1_results/';
    
    % Ensure output folder exists
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % Process each image using its best parameters
    for idx = 1:size(params, 1)
        image_name = params.Image_Name{idx};
        img_path = fullfile(input_folder, image_name);
        
        % Check if the image exists
        if ~isfile(img_path)
            fprintf('Skipping: %s (File not found)\n', img_path);
            continue;
        end

        % Read image
        img = imread(img_path);
        img = im2double(img);  % Convert for precision

        % Convert to YCbCr
        img_ycbcr = rgb2ycbcr(img);
        Y = img_ycbcr(:,:,1);  % Extract luminance channel

        % Get the best parameters for this image
        best_ghe_bins = params.Num_Bins(idx);
        best_ahe_tile = params.Tile_Size(idx);
        best_clahe_tile = params.Tile_Size_1(idx);
        best_clahe_clip = params.Clip_Limit(idx);

        % Apply contrast enhancement using best parameters
        Y_ghe = global_histogram_equalization(Y, best_ghe_bins);
        Y_ahe = adaptive_histogram_equalization(Y, best_ahe_tile);
        Y_clahe = contrast_limited_ahe(Y, best_clahe_tile, best_clahe_clip);

        % Merge enhanced Y channels with original CbCr
        img_ghe = ycbcr2rgb(cat(3, Y_ghe, img_ycbcr(:,:,2), img_ycbcr(:,:,3)));
        img_ahe = ycbcr2rgb(cat(3, Y_ahe, img_ycbcr(:,:,2), img_ycbcr(:,:,3)));
        img_clahe = ycbcr2rgb(cat(3, Y_clahe, img_ycbcr(:,:,2), img_ycbcr(:,:,3)));

        % Compute evaluation metrics
        entropy_ghe = entropy(Y_ghe);
        entropy_ahe = entropy(Y_ahe);
        entropy_clahe = entropy(Y_clahe);

        std_ghe = std(double(Y_ghe(:)));
        std_ahe = std(double(Y_ahe(:)));
        std_clahe = std(double(Y_clahe(:)));

        ssim_ghe = ssim(Y_ghe, Y);
        ssim_ahe = ssim(Y_ahe, Y);
        ssim_clahe = ssim(Y_clahe, Y);

        bhatta_ghe = bhattacharyya_distance(Y, Y_ghe);
        bhatta_ahe = bhattacharyya_distance(Y, Y_ahe);
        bhatta_clahe = bhattacharyya_distance(Y, Y_clahe);

        lcv_ghe = local_contrast_variance(Y_ghe);
        lcv_ahe = local_contrast_variance(Y_ahe);
        lcv_clahe = local_contrast_variance(Y_clahe);

        % % Display the processed images
        % figure;
        % sgtitle(['Contrast Enhancement: ', image_name], 'FontSize', 14);
        % subplot(1, 4, 1), imshow(img), title('Original', 'FontSize', 10);
        % subplot(1, 4, 2), imshow(img_ghe), title(['GHE - Bins: ', num2str(best_ghe_bins)], 'FontSize', 10);
        % subplot(1, 4, 3), imshow(img_ahe), title(['AHE - Tile: ', num2str(best_ahe_tile)], 'FontSize', 10);
        % subplot(1, 4, 4), imshow(img_clahe), title(['CLAHE - Tile: ', num2str(best_clahe_tile), ' Clip: ', num2str(best_clahe_clip)], 'FontSize', 10);

        % Define output folder
        if ~exist(output_folder, 'dir')
            mkdir(output_folder);
        end
        
        % Save Original Image
        original_output = fullfile(output_folder, sprintf('%s_Original.jpg', image_name));
        imwrite(img, original_output);
        
        % Save GHE Image
        ghe_output = fullfile(output_folder, sprintf('%s_GHE.jpg', image_name));
        imwrite(img_ghe, ghe_output);
        
        % Save AHE Image
        ahe_output = fullfile(output_folder, sprintf('%s_AHE.jpg', image_name));
        imwrite(img_ahe, ahe_output);
        
        % Save CLAHE Image
        clahe_output = fullfile(output_folder, sprintf('%s_CLAHE.jpg', image_name));
        imwrite(img_clahe, clahe_output);
        
        disp('All images saved successfully!');

        % Store evaluation metrics in a table
        results_table = table(["GHE"; "AHE"; "CLAHE"], [entropy_ghe; entropy_ahe; entropy_clahe], ...
             [std_ghe; std_ahe; std_clahe], [ssim_ghe; ssim_ahe; ssim_clahe], ...
             [bhatta_ghe; bhatta_ahe; bhatta_clahe], [lcv_ghe; lcv_ahe; lcv_clahe], ...
             'VariableNames', {'Method', 'Entropy', 'StdDev', 'SSIM', 'Bhattacharyya', 'LCV'});

        % Save results table as CSV
        csv_output = fullfile(output_folder, sprintf('Metrics_%s.csv', image_name));
        writetable(results_table, csv_output);
    end
end

% ---- Global Histogram Equalization Function ----
function output = global_histogram_equalization(image, num_bins)
    output = histeq(image, num_bins);
end

% ---- Adaptive Histogram Equalization Function ----
function output = adaptive_histogram_equalization(image, tile_size)
    output = adapthisteq(image, 'NumTiles', [tile_size tile_size]);
end

% ---- Contrast-Limited AHE Function ----
function output = contrast_limited_ahe(image, tile_size, clip_limit)
    output = adapthisteq(image, 'NumTiles', [tile_size tile_size], 'ClipLimit', clip_limit/100);
end

% ---- Bhattacharyya Distance Function ----
function dist = bhattacharyya_distance(img1, img2)
    hist1 = imhist(img1) / numel(img1);
    hist2 = imhist(img2) / numel(img2);
    coeff = sum(sqrt(hist1 .* hist2));
    dist = sqrt(1 - coeff);
end

% ---- Local Contrast Variance (LCV) Function ----
function lcv = local_contrast_variance(image)
    image = im2double(image);
    window_size = 5;
    local_means = imboxfilt(image, window_size);
    local_variance = imboxfilt((image - local_means).^2, window_size);
    lcv = mean(local_variance(:));
end