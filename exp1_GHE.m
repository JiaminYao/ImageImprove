function exp1_GHE()
    % Define image names (Only JPG images)
    image_names = {'D0', 'D1', 'D2', 'D3', 'D4', 'L0', 'L1', 'L2', 'L3', 'L4'};
    input_folder = 'dataset/exp1/';
    output_folder = 'exp1_GHE/';

    % Ensure output folder exists
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % Define histogram bin values to test
    bin_values = [32, 64, 128, 256];

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
        sgtitle(['GHE Parameter Evaluation: ', full_image_name], 'FontSize', 14);

        % Display the original image
        subplot(2,3,1);
        imshow(img);
        title('Original Image', 'FontSize', 10);

        % Loop through different bin values for GHE
        for i = 1:length(bin_values)
            n_bins = bin_values(i);

            % Apply GHE with specific bin size
            Y_ghe = global_histogram_equalization(Y, n_bins);

            % Merge enhanced Y with original CbCr channels
            img_ghe = ycbcr2rgb(cat(3, Y_ghe, img_ycbcr(:,:,2), img_ycbcr(:,:,3)));

            % Compute evaluation metrics
            entropy_ghe = entropy(Y_ghe);
            std_ghe = std(double(Y_ghe(:)));
            ssim_ghe = ssim(Y_ghe, Y);
            bhatta_ghe = bhattacharyya_distance(Y, Y_ghe);
            lcv_ghe = local_contrast_variance(Y_ghe);

            % Store results
            results = [results; n_bins, entropy_ghe, std_ghe, ssim_ghe, bhatta_ghe, lcv_ghe];

            % Display results visually with an updated title
            subplot(2, 3, i+1);
            imshow(img_ghe);
            title(['Bins: ', num2str(n_bins)], 'FontSize', 10);

            % Save processed images in JPG format
            enhanced_img_path = fullfile(output_folder, sprintf('%s_GHE_Bins_%d.jpg', base_name, n_bins));
            imwrite(img_ghe, enhanced_img_path);
        end

        % Create a table with image name and method
        num_rows = size(results, 1);
        image_names_col = repmat({full_image_name}, num_rows, 1);
        methods_col = repmat({'GHE'}, num_rows, 1);

        % Convert results to table and include image name and method
        results_table = table(image_names_col, methods_col, results(:,1), results(:,2), results(:,3), ...
            results(:,4), results(:,5), results(:,6), ...
            'VariableNames', {'Image_Name', 'Method', 'Num_Bins', 'Entropy', 'StdDev', 'SSIM', 'Bhattacharyya', 'LCV'});

        % Append results to all_results
        all_results = [all_results; results_table];

        % Print Image Name and Method Before Table
        fprintf('\n====================================\n');
        fprintf('Image Name: %s\n', full_image_name);
        fprintf('Method: GHE (Global Histogram Equalization)\n');
        fprintf('====================================\n');

        % Display table
        disp('Evaluation Metrics for GHE with Different Bin Values:');
        disp(results_table);

        % Generate metric comparison plots
        plot_ghe_metrics(results_table, bin_values, full_image_name, output_folder);
    end

    % Save final results to a single CSV file
    csv_filename = fullfile(output_folder, 'results_summary.csv');
    writetable(all_results, csv_filename);
    fprintf('All results saved to: %s\n', csv_filename);
end

% ---- Global Histogram Equalization Function ----
function output = global_histogram_equalization(image, num_bins)
    if nargin < 2
        num_bins = 256; % Default to 256 bins
    end
    output = histeq(image, num_bins);
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
function plot_ghe_metrics(results_table, bin_values, image_name, output_folder)
    figure;

    % Entropy Plot
    subplot(2,3,1);
    plot(bin_values, results_table.Entropy(2:end), '-o');
    title('Entropy', 'FontSize', 10);
    xlabel('Number of Bins');
    ylabel('Entropy');
    xticks(bin_values);

    % Standard Deviation Plot
    subplot(2,3,2);
    plot(bin_values, results_table.StdDev(2:end), '-o');
    title('Standard Deviation', 'FontSize', 10);
    xlabel('Number of Bins');
    ylabel('Std Dev');
    xticks(bin_values);

    % SSIM Plot
    subplot(2,3,3);
    plot(bin_values, results_table.SSIM(2:end), '-o');
    title('SSIM', 'FontSize', 10);
    xlabel('Number of Bins');
    ylabel('SSIM');
    xticks(bin_values);

    % Bhattacharyya Distance Plot
    subplot(2,3,4);
    plot(bin_values, results_table.Bhattacharyya(2:end), '-o');
    title('Bhattacharyya Distance', 'FontSize', 10);
    xlabel('Number of Bins');
    ylabel('Distance');
    xticks(bin_values);

    % Local Contrast Variance (LCV) Plot
    subplot(2,3,5);
    plot(bin_values, results_table.LCV(2:end), '-o');
    title('LCV', 'FontSize', 10);
    xlabel('Number of Bins');
    ylabel('LCV');
    xticks(bin_values);

    % Improve layout
    sgtitle(['GHE Parameter Evaluation: ', image_name], 'FontSize', 12);

    % Save the plot in JPG format
    plot_filename = fullfile(output_folder, sprintf('GHE_Bins_Metrics_%s.jpg', image_name));
    saveas(gcf, plot_filename);
end
