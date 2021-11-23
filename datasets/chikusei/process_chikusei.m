
clear all
close all
clc 

% Settings
ratio       = 4;
kernel_size = [8, 8];
sig         = (1/(2*(2.7725887)/ratio^2))^0.5;
start_pos(1)=1;
start_pos(2)=1;

%Load Pavia-Center dataset
load("chikusei.mat")

%Size of the chikusei dataset
[L, W, C] = size(chikusei);

%Creating patches
patch_size = 256;
n_spectral = 128;

%Pavia dataset RGB frequency bands
R=61;
G=35;
B=10;

%Display full pavia image
chikusei_r = chikusei(:,:,R);
chikusei_g = chikusei(:,:,G);
chikusei_b = chikusei(:,:,B);
chikusei_rgb = cat(3, chikusei_r, chikusei_g, chikusei_b);
figure(1);
title('Chikusei Dataset: RGB image: R=40, G=35, B=10')
hold on;
imshow(chikusei_rgb.*5.5/max(chikusei_rgb, [], "all"));

%Apply Bluring before downsampling
BlurKer     = fspecial('gaussian', [kernel_size(1), kernel_size(2)], sig);
chikusei_blur= imfilter(chikusei, BlurKer, 'circular');
Y           = chikusei_blur(start_pos(1):ratio:end, start_pos(2):ratio:end, :);

%Remove noisy values
max_val = round(max(chikusei_blur, [], "all"));
chikusei(chikusei>max_val)=max_val;

%Display blur version of pavia RGB
chikusei_r = chikusei_blur(:,:,R);
chikusei_g = chikusei_blur(:,:,G);
chikusei_b = chikusei_blur(:,:,B);
chikusei_rgb = cat(3, chikusei_r, chikusei_g, chikusei_b);
figure(2);
title('Chikusei Dataset: Blured RGB image: R=60, G=30, B=10')
hold on;
imshow(chikusei_rgb.*5.5/max(chikusei_rgb, [], "all"));

%Compute full pavia pan image
chikusei_pan = mean(chikusei(:,:,60:80), 3);
chikusei_pan = chikusei_pan.*max(chikusei, [], "all")/max(chikusei_pan, [], "all");

% %Saving patches
% count=1;
% mkdir chikusei
% for l = 1:floor(L/patch_size)
%     for w = 1:floor(W/patch_size)
%         % Creating the folder for each patch...
%         folder_name = strcat('chikusei_', num2str(count, '%02d'));
%         file_name= fullfile('./chikusei', folder_name);
%         mkdir(file_name)
%         
%         % Creating reference
%         ref = chikusei((l-1)*patch_size +1: l*patch_size, (w-1)*patch_size +1: w*patch_size, :);
%         %fig1 = disp_rgb(ref);
%         %saveas(fig1, strcat(file_name,'/pavia_ref_', num2str(count, '%02d'),'.png'));
%         
%         % Creating LR HS
%         [y, BC] =downsample(ref,ratio,kernel_size,sig,start_pos);
%         %fig2 = disp_rgb(y);
%         %saveas(fig2, strcat(file_name,'/pavia_lr_', num2str(count, '%02d'),'.png'));
%         
%         % Crating pan image
%         pan = chikusei_pan((l-1)*patch_size +1: l*patch_size, (w-1)*patch_size +1: w*patch_size);
%         
%         % Saving ref, y, pan
%         save(strcat(file_name, '/chikusei_', num2str(count, '%02d'),'.mat'), 'ref', 'y', 'pan');
%         count=count+1;
%     end
% end
% 
% close all;
