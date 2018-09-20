clear;clc;
d = dir('*.tif');
H=0;
W =0;

for ii=1:length(d)
    info = imfinfo(d(ii).name);
    H = max(info.Height,H);
    W = max(info.Width,W);
    
end

for ii=1:length(d)
    if mod(ii,5)==0
        fprintf('Working on %d of %d\n',ii,length(d));
    end
    
    I = imread(d(ii).name);
    I=rgb2gray(I);
    basename = d(ii).name;
    H_i = size(I,1);
    W_i = size(I,2);
    sub_square = I(1:100,1:200);
    rand_range = prctile(double(sub_square(:)),[10,90]);
%     rand_range = [min(min(sub_square)),max(max(sub_square))];
    
    I_out = uint8(randi(rand_range,H,W));

    
    center = round([size(I_out,1)/2,size(I_out,2)/2]);
    H_bot = round(center(1)-H_i(1)/2);
    W_bot = round(center(2)-W_i(1)/2);
    H_bot = max(1,H_bot);
    W_bot = max(1,W_bot);
    H_top = min(H,H_bot-1+H_i);
    W_top = min(W,W_bot-1+W_i);
    
    I_out(H_bot:H_top,W_bot:W_top)= I;
    
    outname = [basename(1:5) 'proc_' basename(6:end)];
    imwrite(I_out,outname);
end 