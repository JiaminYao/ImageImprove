clc; clear; close all;

input_folder = 'dataset/exp3';
output_folder = 'exp3_Selective';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

best_sigmas = containers.Map();
best_sigmas('E0.jpg') = [5,10,20,30,50];
best_sigmas('E1.jpg') = [20,40,60,80,100];
best_sigmas('E2.jpg') = [5,10,20,30,50];
best_sigmas('E3.jpg') = [20,40,60,80,100];
best_sigmas('E4.jpg') = [20,40,60,80,100];
best_sigmas('H0.jpg') = [20,40,60,80,100];
best_sigmas('H1.jpg') = [5,10,20,30,50];
best_sigmas('H2.jpg') = [20,40,60,80,100];
best_sigmas('H3.jpg') = [20,40,60,80,100];
best_sigmas('H4.jpg') = [15,30,50,70,90];

image_names = keys(best_sigmas);

for i = 1:length(image_names)
    img_name = image_names{i};
    sigmas = best_sigmas(img_name);
    
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

    % === Create selective blurred image ===
    num_levels = length(sigmas);
    blurred_layers = cell(3, num_levels);
    for c = 1:3
        for s = 1:num_levels
            blurred_layers{c, s} = imgaussfilt(img(:,:,c), sigmas(s));
        end
    end
    sigma_map = mat2gray(bwdist(dog_mask));
    sigma_idx = round(rescale(sigma_map, 1, num_levels));
    sigma_idx = min(max(sigma_idx, 1), num_levels);

    img_selective = img;
    for c = 1:3
        channel_out = zeros(rows, cols);
        for s = 1:num_levels
            layer = blurred_layers{c, s};
            channel_out(sigma_idx == s) = layer(sigma_idx == s);
        end
        img_selective(:,:,c) = img(:,:,c).*dog_mask + channel_out.*(1 - dog_mask);
    end

    % === Save output ===
    imwrite(img_selective, fullfile(output_folder, ['selective_' img_name]));
    imwrite(dog_mask, fullfile(output_folder, ['mask_' img_name]));

    fprintf('Processed %s with sigmas = [%s]\n', img_name, num2str(sigmas));
end

disp('All images have been processed and saved according to the manual Selective parameters');
