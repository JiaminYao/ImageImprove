% clc; clear; close all;

% Load Image
img = imread('D2.jpg');  % Replace with your image file
img = im2double(img);  % Convert to double for processing

% Increase brightness (add intensity)
alpha = 1.7;  % Brightness scaling factor (>1 for overexposure)
overexposed_img = img * alpha;  % Increase brightness
overexposed_img(overexposed_img > 1) = 1; % Clip values to avoid overflow

% Display original and overexposed image
imshow(overexposed_img); 