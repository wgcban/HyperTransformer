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
load("Pavia_centre.mat")

%Size of the pavia dataset
[L, W, C] = size(pavia);

%Creating patches
patch_size = 160;
n_spectral = 102;

%Pavia dataset RGB frequency bands
R=60;
G=30;
B=10;

%Display full pavia image
pavia_r = pavia(:,:,R);
pavia_g = pavia(:,:,G);
pavia_b = pavia(:,:,B);
pavia_rgb = cat(3, pavia_r, pavia_g, pavia_b);
figure(1);
title('Pavia Centre Dataset: RGB image: R=60, G=30, B=10')
hold on;
imshow(pavia_rgb.*2.5/max(pavia_rgb, [], "all"));

%Apply Bluring before downsampling
BlurKer     = fspecial('gaussian', [kernel_size(1), kernel_size(2)], sig);
pavia_blur  = imfilter(pavia, BlurKer, 'circular');
Y           = pavia_blur(start_pos(1):ratio:end, start_pos(2):ratio:end, :);

%Display blur version of pavia RGB
paviab_r = pavia_blur(:,:,R);
paviab_g = pavia_blur(:,:,G);
paviab_b = pavia_blur(:,:,B);
paviab_rgb = cat(3, paviab_r, paviab_g, paviab_b);
figure(2);
title('Pavia Centre Dataset: Blured RGB image: R=60, G=30, B=10')
hold on;
imshow(paviab_rgb.*2.5/max(paviab_rgb, [], "all"));

%Compute full pavia pan image
pavia_pan = mean(pavia(:,:,1:100), 3); % https://crs.hi.is/?page_id=877

%Saving patches
count=1;
mkdir pavia
for l = 1:floor(L/patch_size)
    for w = 1:floor(W/patch_size)
        % Creating the folder for each patch...
        folder_name = strcat('pavia_', num2str(count, '%02d'));
        file_name= fullfile('./pavia', folder_name);
        mkdir(file_name)
        
        % Creating reference
        ref = pavia((l-1)*patch_size +1: l*patch_size, (w-1)*patch_size +1: w*patch_size, :);
        %fig1 = disp_rgb(ref);
        %saveas(fig1, strcat(file_name,'/pavia_ref_', num2str(count, '%02d'),'.png'));
        
        % Creating LR HS
        [y, BC] =downsample(ref,ratio,kernel_size,sig,start_pos);
        %fig2 = disp_rgb(y);
        %saveas(fig2, strcat(file_name,'/pavia_lr_', num2str(count, '%02d'),'.png'));
        
        % Crating pan image
        pan = pavia_pan((l-1)*patch_size +1: l*patch_size, (w-1)*patch_size +1: w*patch_size);
        
        % Saving ref, y, pan
        save(strcat(file_name, '/pavia_', num2str(count, '%02d'),'.mat'), 'ref', 'y', 'pan');
        count=count+1;
    end
end

close all;
% ref_mean = squeeze(mean(pavia, [1,2]))';
% writematrix(ref_mean,'pavia_ref_mean.txt','Delimiter','comma');
% ref_std = squeeze(std(pavia, 0, [1,2]))';
% writematrix(ref_std,'pavia_ref_std.txt','Delimiter','comma');
% 
% Y_mean = squeeze(mean(Y, [1,2]))';
% writematrix(Y_mean,'pavia_Y_mean.txt','Delimiter','comma');
% Y_std = squeeze(std(Y, 0, [1,2]))';
% writematrix(Y_std,'pavia_Y_std.txt','Delimiter','comma');
