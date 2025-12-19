clc; clear; close all;

input_folder = 'dataset/exp3/';
output_folder = 'exp3_Luminance/';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% format：image_name => sigma
best_sigma = containers.Map();
best_sigma('E0.jpg') = 5;
best_sigma('E1.jpg') = 5;
best_sigma('E2.jpg') = 5;
best_sigma('E3.jpg') = 5;
best_sigma('E4.jpg') = 5;
best_sigma('H0.jpg') = 5;
best_sigma('H1.jpg') = 5;
best_sigma('H2.jpg') = 5;
best_sigma('H3.jpg') = 5;
best_sigma('H4.jpg') = 5;

image_files = keys(best_sigma);
for i = 1:length(image_files)
    img_name = image_files{i};
    sigma = best_sigma(img_name);

    % === Load image ===
    img = im2double(imread(fullfile(input_folder, img_name)));
    if size(img, 3) == 3
        lum = rgb2gray(img);
    else
        lum = img;
    end
    [rows, cols, ~] = size(img);

    % === Compute inverse luminance and blur weight ===
    inv_lum = 1 - mat2gray(lum);
    blur_weight = imgaussfilt(inv_lum, sigma);         % smooth weight map
    blurred = imgaussfilt(img, sigma);                 % globally blurred image

    % === Apply luminance-guided blending ===
    img_luminance = img;
    for c = 1:3
        img_luminance(:,:,c) = img(:,:,c) .* (1 - blur_weight) + blurred(:,:,c) .* blur_weight;
    end

    % === Save result ===
    imwrite(img_luminance, fullfile(output_folder, ['luminance_' img_name]));
    fprintf('Processed %s with sigma = %.1f (luminance-guided blur)\n', img_name, sigma);
end

disp('✅ All luminance-guided images processed and saved.');
