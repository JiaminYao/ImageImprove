function exp1_CLAHE()
    % Define image names
    image_names = {'D0', 'D1', 'D2', 'D3', 'D4', 'L0', 'L1', 'L2', 'L3', 'L4'};
    input_folder = 'dataset/exp1/';
    output_folder = 'exp1_CLAHE/';

    % Ensure output folder exists
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % Define CLAHE parameters
    tile_sizes = [8, 16, 32];
    clip_limits = [2, 4, 6, 8];

    % Initialize results table
    all_results = [];

    % Process each image (JPG only)
    for idx = 1:length(image_names)
        image_name = image_names{idx};
        img_path = fullfile(input_folder, [image_name, '.jpg']);

        % Check if image exists
        if ~isfile(img_path)
            fprintf('Skipping: %s (File not found)\n', img_path);
            continue;
        end

        img = imread(img_path);
        [~, base_name, ext] = fileparts(img_path); % Extract filename
        full_image_name = [base_name, ext];

        % Convert image to double for precision
        img = im2double(img);

        % Convert image to YCbCr color space
        img_ycbcr = rgb2ycbcr(img);
        Y = img_ycbcr(:,:,1);  % Extract luminance channel

        % Store results
        results = [];

        % Original Image Metrics
        entropy_original = entropy(Y);
        std_original = std(double(Y(:)));
        ssim_original = 1.0;  % SSIM with itself is always 1
        bhatta_original = 0.0; % Bhattacharyya distance with itself is 0
        lcv_original = local_contrast_variance(Y);

        % Store original image metrics
        results = [results; 0, 0, entropy_original, std_original, ssim_original, bhatta_original, lcv_original];

        % Create figure for image results
        figure;
        subplot(4, 5, 1);
        imshow(img);
        title('Original Image', 'FontSize', 10);

        % Loop through different tile sizes and clip limits
        count = 2;
        for i = 1:length(tile_sizes)
            for j = 1:length(clip_limits)
                tile_size = tile_sizes(i);
                clip_limit = clip_limits(j);

                % Apply CLAHE with specific tile size and clip limit
                Y_clahe = contrast_limited_ahe(Y, tile_size, clip_limit);

                % Merge enhanced Y with original CbCr channels
                img_clahe = ycbcr2rgb(cat(3, Y_clahe, img_ycbcr(:,:,2), img_ycbcr(:,:,3)));

                % Compute evaluation metrics
                entropy_clahe = entropy(Y_clahe);
                std_clahe = std(double(Y_clahe(:)));
                ssim_clahe = ssim(Y_clahe, Y);
                bhatta_clahe = bhattacharyya_distance(Y, Y_clahe);
                lcv_clahe = local_contrast_variance(Y_clahe);

                % Store results
                results = [results; tile_size, clip_limit, entropy_clahe, std_clahe, ssim_clahe, bhatta_clahe, lcv_clahe];

                % Show processed images
                subplot(4, 5, count);
                imshow(img_clahe);
                title(sprintf('T: %d, C: %d', tile_size, clip_limit), 'FontSize', 8);
                count = count + 1;

                % Save processed images in JPG format
                enhanced_img_path = fullfile(output_folder, sprintf('%s_CLAHE_Tile_%d_Clip_%d.jpg', base_name, tile_size, clip_limit));
                imwrite(img_clahe, enhanced_img_path);
            end
        end

        % Save image results figure
        sgtitle(['CLAHE Image Results: ', full_image_name], 'FontSize', 12);
        img_plot_filename = fullfile(output_folder, sprintf('CLAHE_Images_%s.jpg', full_image_name));
        saveas(gcf, img_plot_filename);

        % Create a table with image name and method
        num_rows = size(results, 1);
        image_names_col = repmat({full_image_name}, num_rows, 1);
        methods_col = repmat({'CLAHE'}, num_rows, 1);

        % Convert results to table and include image name and method
        results_table = table(image_names_col, methods_col, results(:,1), results(:,2), results(:,3), ...
            results(:,4), results(:,5), results(:,6), results(:,7), ...
            'VariableNames', {'Image_Name', 'Method', 'Tile_Size', 'Clip_Limit', 'Entropy', 'StdDev', 'SSIM', 'Bhattacharyya', 'LCV'});

        % Append results to all_results
        all_results = [all_results; results_table];

        % Generate a single Heatmap plot for all five metrics
        plot_clahe_heatmaps(results_table, tile_sizes, clip_limits, full_image_name, output_folder);
    end

    % Save final results to a single CSV file
    csv_filename = fullfile(output_folder, 'results_summary.csv');
    writetable(all_results, csv_filename);
    fprintf('All results saved to: %s\n', csv_filename);
end

% ---- CLAHE Function ----
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

% ---- Local Contrast Variance Function (FIXED) ----
function lcv = local_contrast_variance(image)
    image = im2double(image);
    window_size = 5;
    local_means = imboxfilt(image, window_size);
    local_variance = imboxfilt((image - local_means).^2, window_size);
    lcv = mean(local_variance(:));
end

% ---- Heatmap Plot for CLAHE ----
function plot_clahe_heatmaps(results_table, tile_sizes, clip_limits, image_name, output_folder)
    metrics = {'Entropy', 'StdDev', 'SSIM', 'Bhattacharyya', 'LCV'};

    figure;
    for k = 1:length(metrics)
        metric = metrics{k};
        Z = reshape(results_table{2:end, metric}, length(tile_sizes), length(clip_limits));

        subplot(2, 3, k);
        h = heatmap(clip_limits, tile_sizes, Z);
        h.XLabel = 'Clip Limit';
        h.YLabel = 'Tile Size';
        h.Title = metric;
        colormap jet;
        colorbar;
    end

    % Save the combined plot
    sgtitle(['CLAHE Evaluation: ', image_name], 'FontSize', 12);
    plot_filename = fullfile(output_folder, sprintf('CLAHE_Metrics_%s.jpg', image_name));
    saveas(gcf, plot_filename);
end
