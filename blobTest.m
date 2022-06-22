detector = vision.ForegroundDetector(...
       'NumTrainingFrames', 10, 'LearningRate', 0.05,...
       'InitialVariance', 30*30);
%blob = vision.BlobAnalysis('Connectivity',8,...
%       'CentroidOutputPort', false, 'AreaOutputPort', false, ...
%       'BoundingBoxOutputPort', true, ...
%       'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 100);
   
blob = vision.BlobAnalysis('Connectivity',8,...
       'CentroidOutputPort', false, 'AreaOutputPort', false, ...
       'BoundingBoxOutputPort', true, ...
       'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 100);
   
shapeInserter = vision.ShapeInserter('BorderColor','White');
window_sz = floor(target_sz * (1 + padding));
figure
for frame = 1:20%numel(img_files) 
im = imread([video_path img_files{frame}]);
im = rgb2gray(im);
mask = [];
mask = detector(im);
mask = bwmorph(mask, 'close',10);
mask = bwmorph(mask, 'fill',10);
bbox=[];
bbox   = blob(mask);
out    = shapeInserter(im,bbox);

imshow(out)
end