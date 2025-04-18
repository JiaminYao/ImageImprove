function exp1_AHE()
    % Define image names (Only JPG images)
    image_names = {'D0', 'D1', 'D2', 'D3', 'D4', 'L0', 'L1', 'L2', 'L3', 'L4'};
    input_folder = 'dataset/exp1/';
    output_folder = 'exp1_AHE/';

    % Ensure output folder exists
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % Define AHE tile sizes to test
    tile_sizes = [8, 16, 32];

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
        results = [results; 0, entropy_original, std_original, ssim_original, bhatta_original, lcv_original];

        % Create figure with a full title
        figure;
        sgtitle(['AHE Parameter Evaluation: ', full_image_name], 'FontSize', 14);

        % Display the original image
        subplot(2,3,1);
        imshow(img);
        title('Original Image', 'FontSize', 10);

        % Loop through different tile sizes for AHE
        for i = 1:length(tile_sizes)
            tile_size = tile_sizes(i);

            % Apply AHE with specific tile size
            Y_ahe = adaptive_histogram_equalization(Y, tile_size);

            % Merge enhanced Y with original CbCr channels
            img_ahe = ycbcr2rgb(cat(3, Y_ahe, img_ycbcr(:,:,2), img_ycbcr(:,:,3)));

            % Compute evaluation metrics
            entropy_ahe = entropy(Y_ahe);
            std_ahe = std(double(Y_ahe(:)));
            ssim_ahe = ssim(Y_ahe, Y);
            bhatta_ahe = bhattacharyya_distance(Y, Y_ahe);
            lcv_ahe = local_contrast_variance(Y_ahe);

            % Store results
            results = [results; tile_size, entropy_ahe, std_ahe, ssim_ahe, bhatta_ahe, lcv_ahe];

            % Display results visually with an updated title
            subplot(2, 3, i+1);
            imshow(img_ahe);
            title(['Tile: ', num2str(tile_size)], 'FontSize', 10);

            % Save processed images in JPG format
            enhanced_img_path = fullfile(output_folder, sprintf('%s_AHE_Tile_%d.jpg', base_name, tile_size));
            imwrite(img_ahe, enhanced_img_path);
        end

        % Create a table with image name and method
        num_rows = size(results, 1);
        image_names_col = repmat({full_image_name}, num_rows, 1);
        methods_col = repmat({'AHE'}, num_rows, 1);

        % Convert results to table and include image name and method
        results_table = table(image_names_col, methods_col, results(:,1), results(:,2), results(:,3), ...
            results(:,4), results(:,5), results(:,6), ...
            'VariableNames', {'Image_Name', 'Method', 'Tile_Size', 'Entropy', 'StdDev', 'SSIM', 'Bhattacharyya', 'LCV'});

        % Append results to all_results
        all_results = [all_results; results_table];

        % Print Image Name and Method Before Table
        fprintf('\n====================================\n');
        fprintf('Image Name: %s\n', full_image_name);
        fprintf('Method: AHE (Adaptive Histogram Equalization)\n');
        fprintf('====================================\n');

        % Display table
        disp('Evaluation Metrics for AHE with Different Tile Sizes:');
        disp(results_table);

        % Generate metric comparison plots
        plot_ahe_metrics(results_table, tile_sizes, full_image_name, output_folder);
    end

    % Save final results to a single CSV file
    csv_filename = fullfile(output_folder, 'results_summary.csv');
    writetable(all_results, csv_filename);
    fprintf('All results saved to: %s\n', csv_filename);
end

% ---- Adaptive Histogram Equalization Function ----
function output = adaptive_histogram_equalization(image, tile_size)
    output = adapthisteq(image, 'NumTiles', [tile_size tile_size]);
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

% ---- Visualization Function ----
function plot_ahe_metrics(results_table, tile_sizes, image_name, output_folder)
    hFig = figure;

    % Entropy Plot
    subplot(2,3,1);
    plot(tile_sizes, results_table.Entropy(2:end), '-o');
    title('Entropy', 'FontSize', 10);
    xlabel('Tile Size');
    ylabel('Entropy');
    xticks(tile_sizes);

    % Standard Deviation Plot
    subplot(2,3,2);
    plot(tile_sizes, results_table.StdDev(2:end), '-o');
    title('Standard Deviation', 'FontSize', 10);
    xlabel('Tile Size');
    ylabel('Std Dev');
    xticks(tile_sizes);

    % SSIM Plot
    subplot(2,3,3);
    plot(tile_sizes, results_table.SSIM(2:end), '-o');
    title('SSIM', 'FontSize', 10);
    xlabel('Tile Size');
    ylabel('SSIM');
    xticks(tile_sizes);

    % Bhattacharyya Distance Plot
    subplot(2,3,4);
    plot(tile_sizes, results_table.Bhattacharyya(2:end), '-o');
    title('Bhattacharyya Distance', 'FontSize', 10);
    xlabel('Tile Size');
    ylabel('Distance');
    xticks(tile_sizes);

    % Local Contrast Variance (LCV) Plot
    subplot(2,3,5);
    plot(tile_sizes, results_table.LCV(2:end), '-o');
    title('LCV', 'FontSize', 10);
    xlabel('Tile Size');
    ylabel('LCV');
    xticks(tile_sizes);

    % Improve layout
    sgtitle(['AHE Parameter Evaluation: ', image_name], 'FontSize', 12);

    % Save the plot in JPG format
    plot_filename = fullfile(output_folder, sprintf('AHE_Tile_Metrics_%s.jpg', image_name));
    saveas(hFig, plot_filename);  % Use the figure handle
    close(hFig);  % Optional: close the figure after saving
end