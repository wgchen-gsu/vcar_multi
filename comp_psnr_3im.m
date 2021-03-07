function comp_psnr_3im()
FolderA = fullfile('output_A/');
imfile  = dir(fullfile(FolderA, '*.png'));
num_im  = size(imfile, 1);

ssimV   = [];
peaksnr = [];
for(n = 1 :  num_im)
    name   = imfile(n).name;
    cmpFile= sprintf('%s%s', FolderA, name);
            
    xim = imread(cmpFile);
        
    [height, width, d] = size(xim);
    cols = width/3;
    oim = xim(:,1:cols,1);
    oim = oim(16:height-15, 16:cols-15);
    dim = xim(:, cols+1:cols*2,1);
    dim = dim(16:height-15, 16:cols-15);
    rim = xim(:, cols*2+1:cols*3,1);
    rim = rim(16:height-15, 16:cols-15);
    
    peaksnr = [peaksnr; psnr(oim, rim), psnr(oim, dim)];        
    ssimV   = [ssimV; ssim(oim, rim), ssim(oim, dim)];
end
disp([mean(peaksnr, 1)]);
disp([mean(ssimV, 1)]);

mean(peaksnr(:,1)) - mean(peaksnr(:,2))
mean(ssimV(:,1)) - mean(ssimV(:,2))
return