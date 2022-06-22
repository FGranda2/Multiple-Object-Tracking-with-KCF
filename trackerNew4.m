function [time] = trackerNew4(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization, delete_lostTracks)
    
    Size = size(pos);
    %Condition to resize
	resize_image = (sqrt(prod(target_sz(1,1:2))) >= 100);  %diagonal size >= threshold
    resize_image = 0;

    %For Background Detection
    detector = vision.ForegroundDetector(...
       'NumTrainingFrames', 10, 'LearningRate', 0.05,...
       'InitialVariance', 30*30);
   
    blob = vision.BlobAnalysis('Connectivity',8,...
       'CentroidOutputPort', true, 'AreaOutputPort', false, ...
       'BoundingBoxOutputPort', true, ...
       'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 150);
   
    %Start visualization interface
    if show_visualization
        
		update_visualization1 = show_videoNew(img_files, video_path, resize_image, Size(1,1));

    end
    
    %Measurements
    time = 0;  %to calculate FPS
    if Size(1,1) == 1
            counter = [0,0];

    else
            counter = zeros(Size(1,1),2);
    end
    
    ending = Size(1,1);
    TC = 10;
    %Start Frame by Frame Analysis
	for frame = 1:numel(img_files)
        
        for i = 1:ending
           
            if counter(i,1) > TC || counter(i,2) > TC
               
                window_sz(i) = [];
                output_sigma(i) = [];
                yf(i) = [];
                cos_window(i) = [];
                zf1(i) = [];
                kzf1(i) = [];
                pos(i,:) = [];
                target_sz(i,:) = [];
                box(i,:) = [];
                xf1(i) = [];
                kf1(i) = [];
                alphaf1(i) = [];
                model_alphaf1(i) = [];
                model_xf1(i) = [];
                ending = ending - 1;
                counter(i,:) = [0,0];
                %if ending == 1
                %    counter(i) = [];
                %else
                %    counter = zeros(Size(1,1),2);
                %end
                
                
            end
            
        end
        
        %Load image
		im = imread([video_path img_files{frame}]);
        im = rgb2gray(im);
        
        %If resize needed
		if resize_image
			im = imresize(im, 0.5);
        end
        
        tic() %Start timer for measurements
    
        for i = 1: ending
            
            if frame == 1
                %window size, taking padding into account
                window_sz{i,1,:} = floor(target_sz(i,1:2) * (1 + padding));   
                output_sigma{i,1} = sqrt(prod(target_sz(i,1:2))) * output_sigma_factor / cell_size;
                yf{i,1,:} = fft2(gaussian_shaped_labels(output_sigma{i,1}, floor(window_sz{i,1} / cell_size)));

                %store pre-computed cosine window
                cos_window{i,1,:} = hann(size(yf{i,1},1)) * hann(size(yf{i,1},2))';
            end
            
            if frame > 1
                %obtain a subwindow for detection and transform to F domain
                patch1 = get_subwindow(im, pos(i,1:2), window_sz{i,1});
                zf1{i,1,:} = fft2(get_features(patch1, features, cell_size, cos_window{i,1}));

                %calculate response of the classifier at all shifts
                kzf1{i,1,:} = gaussian_correlation(zf1{i,1}, model_xf1{i,1}, kernel.sigma);
                response1 = real(ifft2(model_alphaf1{i,1} .* kzf1{i,1}));

                %target location is at the maximum response. we must take into
                [vert_delta1, horiz_delta1] = find(response1 == max(response1(:)), 1);

                if vert_delta1 > size(zf1{i,1},1) / 2
                    vert_delta1 = vert_delta1 - size(zf1{i,1},1);
                end
                
                if horiz_delta1 > size(zf1{i,1},2) / 2
                    horiz_delta1 = horiz_delta1 - size(zf1{i,1},2);
                end
                
                %Update estimated new position
                pos(i,1:2) = pos(i,1:2) + cell_size * [vert_delta1 - 1, horiz_delta1 - 1];
                
            end
        end
        
        %KCF boxes creation
        for i=1:ending
                
            ppos = pos(i,1:2);
            ttarget_sz = target_sz(i,1:2);
            box(i,1:4) = [ppos([2,1]) - ttarget_sz([2,1])/2, ttarget_sz([2,1])];

        end
        
        %Morphological operations
        bbox=[];
        cents = [];
        posBoxes = [];
        ttarget_sz = [];
        centroids = [];
        rateSize = 0.01;
        mask = detector(im);
        mask = bwmorph(mask, 'close',50);
        mask = bwmorph(mask, 'fill',50);
        [cents, posBoxes] = blob(mask);
        cents =  round(cents);
        Size2 = size(cents);
        
        %Blob boxes creation with same dimensions
        for i=1:Size2(1,1)

            ppos(1,1) = cents(i,2);
            ppos(1,2) = cents(i,1);
            centroids(i,1:2) = ppos;
            ttarget_sz = target_sz(1,1:2);
            bbox(i,1:4) = [ppos([2,1]) - ttarget_sz([2,1])/2, ttarget_sz([2,1])];

        end
        
        %Overlap Ratio and best match
        if isempty(bbox) == 0
            overlapRatio = bboxOverlapRatio(box,bbox);
            for i = 1:ending

                [match(i,2),match(i,1)] = max(overlapRatio(i,:));
                if match(i,2) > 0.4

                    %box(i,1:4) = bbox(match(i,1), 1:4);
                    posNew(i,1:2) = centroids(match(i,1), 1:2);
                    NewSize(i,1:2)= posBoxes(match(i,1), 3:4);

                else

                    posNew(i,1:2) = [0,0];
                    NewSize(i,1:2) = [0,0];

                end
            end

        else

            match = [];
            if ending == 1
                posNew = [0,0];
                
            else
                posNew = zeros(ending,2);
            end
            
            NewSize = [];

        end
        
        %Update position with weighted mean
        for i = 1: ending
            
            if posNew(i,1) > 0

                pos(i,1) = (pos(i,1) + match(i,2)*posNew(i,1)) / (1+match(i,2));
                target_sz(i,1) = (target_sz(i,1) + rateSize*NewSize(i,2)) / (1+rateSize);
                
            end
            
            if posNew(i,2) > 0
                
                pos(i,2) = (pos(i,2) + match(i,2)*posNew(i,2)) / (1+match(i,2));
                target_sz(i,2) = (target_sz(i,2) + rateSize*NewSize(i,1)) / (1+rateSize);
                
            end
            
            if delete_lostTracks == 1
                if pos(i,1) < 0

                    counter(i,1) = counter(i,1) + 1;
                end

                if pos(i,2) < 0

                    counter(i,2) = counter(i,2) + 1;
                end
            end
            
            %obtain a subwindow for training at newly estimated target position
            patch1 = get_subwindow(im, pos(i,1:2), window_sz{i,1});
            xf1{i,1,:} = fft2(get_features(patch1, features, cell_size, cos_window{i,1}));

            %Kernel Ridge Regression, calculate alphas (in Fourier domain)
            kf1{i,1,:} = gaussian_correlation(xf1{i,1}, xf1{i,1}, kernel.sigma);
            alphaf1{i,1,:} = yf{i,1} ./ (kf1{i,1} + lambda);   %equation for fast training

            if frame == 1  %first frame, train with a single image
                model_alphaf1{i,1} = alphaf1{i,1};
                model_xf1{i,1} = xf1{i,1};
            else
                %subsequent frames, interpolate model
                model_alphaf1{i,1} = (1 - interp_factor) * model_alphaf1{i,1} + interp_factor * alphaf1{i,1};
                model_xf1{i,1} = (1 - interp_factor) * model_xf1{i,1} + interp_factor * xf1{i,1};
            end
            
            
        end
        
        time = time + toc(); %Time measurements
        
        %Boxes creation and visualization
		if show_visualization
            stop1 = update_visualization1(frame, box, ending);
            
			%Stop Mechanics
			if stop1
                break
            end
            
            %Draw updated frame
            drawnow
            
        end
		%F(frame) = getframe(gcf);
        %writeVideo(v,F(frame));
    end
    %close(v);
end