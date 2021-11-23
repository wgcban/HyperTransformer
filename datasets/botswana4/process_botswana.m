% Written by: Chaminda Bandara (wbandar1@jhu.edu/ chaminda.bandara@eng.pdn.ac.lk)
% Date: June-16-2021
% This function process the botswana dataset and generates ref, y, pan patches
% in seperate folders.

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
load("Botswana.mat")
Botswana(Botswana>=7365) = 7365;


%Size of the pavia dataset
[L, W, C] = size(Botswana);

%Creating patches
patch_size = 120;
n_spectral = 120;

%Botswana dataset RGB frequency bands
R = 29;
G = 20;
B = 12;

%Display full pavia image
Botswana_r = Botswana(:,:,R);
Botswana_g = Botswana(:,:,G);
Botswana_b = Botswana(:,:,B);
Botswana_rgb = cat(3, Botswana_r, Botswana_g, Botswana_b);
figure(1);
title('Botswana Dataset: RGB image: R=29, G=20, B=12')
hold on;
imshow(Botswana_rgb.*1.5/max(Botswana_rgb, [], "all"));

%Apply Bluring before downsampling
BlurKer         = fspecial('gaussian', [kernel_size(1), kernel_size(2)], sig);
Botswana_blur   = imfilter(Botswana, BlurKer, 'circular');
Y               = Botswana_blur(start_pos(1):ratio:end, start_pos(2):ratio:end, :);

%Display blur version of pavia RGB
Botswanab_r = Botswana_blur(:,:,R);
Botswanab_g = Botswana_blur(:,:,G);
Botswanab_b = Botswana_blur(:,:,B);
Botswanab_rgb = cat(3, Botswanab_r, Botswanab_g, Botswanab_b);
figure(2);
title('Botswana Dataset: Blured RGB image: R=29, G=20, B=12')
hold on;
imshow(Botswanab_rgb.*1.5/max(Botswanab_rgb, [], "all"));

%Compute full pavia pan image
Botswana_pan = mean(Botswana(:,:,1:29), 3);

%Saving patches
count=1;
mkdir botswana4
for l = 1:floor(L/patch_size)
    for w = 1:floor(W/patch_size)
        % Creating the folder for each patch...
        folder_name = strcat('botswana4_', num2str(count, '%02d'));
        file_name= fullfile('./botswana4', folder_name);
        mkdir(file_name)
        
        % Creating reference
        ref = Botswana((l-1)*patch_size +1: l*patch_size, (w-1)*patch_size +1: w*patch_size, :);
        %fig1 = disp_rgb(ref);
        %saveas(fig1, strcat(file_name,'/botswana_ref_', num2str(count, '%02d'),'.png'));
        
        % Creating LR HS
        [y, BC] =downsample(ref,ratio,kernel_size,sig,start_pos);
        %fig2 = disp_rgb(y);
        %saveas(fig2, strcat(file_name,'/botswana_lr_', num2str(count, '%02d'),'.png'));
        
        % Crating pan image
        pan = Botswana_pan((l-1)*patch_size +1: l*patch_size, (w-1)*patch_size +1: w*patch_size);
        
        % Saving ref, y, pan
        save(strcat(file_name, '/botswana4_', num2str(count, '%02d'),'.mat'), 'ref', 'y', 'pan');
        count=count+1;
    end
end
