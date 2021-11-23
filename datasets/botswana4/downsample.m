function [Y,BluKer]=downsample(X,ratio,size_kernel,sig,start_pos)
    % Input: 
    % X:            reference image, 
    % ratio:        downsampling factor,
    % size_kernel:  size_kernel(1) and size_kernel(2) are the height and length
    % of the blurring kernel
    % 
    % Output: 
    % Y:      blurred and downsampled image
    % BluKer: the blurring kernel
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % generating the kernel
    BluKer = fspecial('gaussian',[size_kernel(1) size_kernel(2)],sig);
    Y=imfilter(X, BluKer, 'circular');
    % Downsample the image: sample one pixel from every 'ratio' pixels
    % Y=Y(1:ratio:end, 1:ratio:end,:);
    Y=Y(start_pos(1):ratio:end, start_pos(2):ratio:end,:);


    % % the size of the kernel
    % len_hor=size_kernel(1);
    % len_ver=size_kernel(2);
    % [nr,nc,L] = size(X);
    % % define convolution operator
    % mid_col=round((nc+1)/2);
    % mid_row=round((nr+1)/2);
    % lx = (len_hor-1)/2;
    % ly = (len_ver-1)/2;
    % % circularly center
    % B=zeros(nr,nc);
    % % range of the pixels
    % B(mid_row-lx:mid_row+lx,mid_col-ly:mid_col+ly)=BluKer;
    % B=circshift(B,[-mid_row+1,-mid_col+1]);
    % %normalize
    % B=B/sum(sum(B));
    % 
    % % FFT of the blurring kernel
    % FB  = repmat(fft2(B),[1 1 L]);
    % ConvCBD = @(X,FK) real(ifft2(fft2(X).*FK)); 
    % % Blur the image with convolution kernel
    % Y=ConvCBD(X,FB);
end
