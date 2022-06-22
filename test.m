close all
clear

%Path to files
base_path = './data/Benchmark/';
video = 'choose';
kernel_type = 'gaussian'; 
feature_type = 'hog';
show_visualization = ~strcmp(video, 'all'); 
show_plots = ~strcmp(video, 'all');

%parameters based on the chosen kernel or feature type
kernel.type = kernel_type;
features.hog = true;
features.gray = false;
padding = 1.5;  %extra area surrounding the target
lambda = 1e-4;  %regularization
output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)
interp_factor = 0.02;
kernel.sigma = 0.5;
kernel.poly_a = 1;
kernel.poly_b = 9;
features.hog_orientations = 9;
cell_size = 4;

%Choose sequence
video = choose_video(base_path);
%Load images
[img_files, pos, target_sz, video_path] = load_video_info(base_path, video);

%Choose objects
im = imread([video_path img_files{1}]);
im = rgb2gray(im);
imshow(im);

prompt = {'Enter number of objects to track:'};
dlgtitle = 'KCF TRACKER';
dims = [1 50];
definput = {'1'};
objects = inputdlg(prompt,dlgtitle,dims,definput);
objects = str2double(objects{1,1});

for i = 1:objects

    rect(i,1:4) = getrect2;
    rectangle('Position', rect(i,1:4),...
    'EdgeColor','g', 'LineWidth', 2)
    hold on
end
close
for i = 1:objects

    rect(i,1) = round(rect(i,1) + (rect(i,3)/2));
    rect(i,2) = round(rect(i,2) + (rect(i,4)/2));
    pos2(i,1) = rect(i,2);
    pos2(i,2) = rect(i,1);
    target_sz2(i,1) = rect(i,4);
    target_sz2(i,2) = rect(i,3);

end
 

%call tracker function with all the relevant parameters
time = trackerNew4(video_path, img_files, pos2, target_sz2, ...
			padding, kernel, lambda, output_sigma_factor, interp_factor, ...
			cell_size, features, show_visualization,0);
        
fps = numel(img_files) / time;