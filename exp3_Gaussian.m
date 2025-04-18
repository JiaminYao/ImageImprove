clc; clear; close all;

input_folder = 'dataset/exp3';
output_folder = 'exp3_Gaussian';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end


% format：image_name => sigma
best_sigma = containers.Map();
best_sigma('E0.jpg') = 5;
best_sigma('E1.jpg') = 15;
best_sigma('E2.jpg') = 5;
best_sigma('E3.jpg') = 20;
best_sigma('E4.jpg') = 20;
best_sigma('H0.jpg') = 20;
best_sigma('H1.jpg') = 5;
best_sigma('H2.jpg') = 20;
best_sigma('H3.jpg') = 10;
best_sigma('H4.jpg') = 20;


image_files = keys(best_sigma);
for i = 1:length(image_files)
    img_name = image_files{i};
    sigma = best_sigma(img_name);
    
    % === Load image ===
    img = im2double(imread(fullfile(input_folder, img_name)));
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    [rows, cols, ~] = size(img);
    
    % === Generate DOG mask ===
    T = graythresh(img_gray);
    mask = imbinarize(img_gray, T);
    if mean(img_gray(mask)) < mean(img_gray(~mask)), mask = ~mask; end
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
    
    % === Apply Gaussian blur on background ===
    blurred = imgaussfilt(img, sigma);
    img_gaussian = img;
    for c = 1:3
        img_gaussian(:,:,c) = img(:,:,c).*dog_mask + blurred(:,:,c).*(1 - dog_mask);
    end
    
    % === Save output ===
    imwrite(img_gaussian, fullfile(output_folder, ['gaussian_' img_name]));
    imwrite(dog_mask, fullfile(output_folder, ['mask_' img_name]));
    
    fprintf('Processed %s with sigma = %.1f\n', img_name, sigma);
end

disp('All images have been processed and saved according to the Gaussian parameters');
