function [positions, time] = trackerNew(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)
    
    matches = {};
    detector = vision.ForegroundDetector(...
       'NumTrainingFrames', 10, 'LearningRate', 0.05,...
       'InitialVariance', 30*30);
   
    blob = vision.BlobAnalysis('Connectivity',8,...
       'CentroidOutputPort', true, 'AreaOutputPort', false, ...
       'BoundingBoxOutputPort', false, ...
       'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 100);
    
    %Condition to resize
	resize_image = (sqrt(prod(target_sz(1,1:2))) >= 100);  %diagonal size >= threshold
	
    %if resize_image
	%	pos = floor(pos / 2);
	%	target_sz = floor(target_sz / 2);
	%end


	%window size, taking padding into account
	window_sz = floor(target_sz(1,1:2) * (1 + padding));   
	output_sigma = sqrt(prod(target_sz(1,1:2))) * output_sigma_factor / cell_size;
    
    if isfield(features, 'deep') && features.deep
        yf = fft2(gaussian_shaped_labels(output_sigma, ceil(window_sz / cell_size)));

    else
        yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
    end

	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
	
	Size = size(pos);   %will later change place
	if show_visualization  %create video interface
		update_visualization1 = show_videoNew(img_files, video_path, resize_image, Size(1,1));
	end
	

	%time = 0;  %to calculate FPS
	%positions = zeros(numel(img_files), 2);  %to calculate precision
    
    
	for frame = 1:150%numel(img_files)
        
        %load image
		im = imread([video_path img_files{frame}]);
        if ~isfield(features, 'deep')
            if size(im,3) > 1
                im = rgb2gray(im);
            end
        end
        
		if resize_image
			im = imresize(im, 0.5);
		end

		tic()
        for i = 1: Size(1,1)
            if frame > 1
                %obtain a subwindow for detection at the position from last
                %frame, and convert to Fourier domain (its size is unchanged)
                patch1 = get_subwindow(im, pos(i,1:2), window_sz);
                zf1{i,1,:} = fft2(get_features(patch1, features, cell_size, cos_window));

                %calculate response of the classifier at all shifts
                switch kernel.type
                case 'gaussian'
                    kzf1 = gaussian_correlation(zf1{i,1}, model_xf1{i,1}, kernel.sigma);
                case 'polynomial'
                    kzf1 = polynomial_correlation(zf1, model_xf1, kernel.poly_a, kernel.poly_b);
                case 'linear'
                    kzf1 = linear_correlation(zf1, model_xf1);
                end
                response1 = real(ifft2(model_alphaf1{i,1} .* kzf1));  %equation for fast detection

                %target location is at the maximum response. we must take into
                %account the fact that, if the target doesn't move, the peak
                %will appear at the top-left corner, not at the center (this is
                %discussed in the paper). the responses wrap around cyclically.
                [vert_delta1, horiz_delta1] = find(response1 == max(response1(:)), 1);

                if vert_delta1 > size(zf1{i,1},1) / 2  %wrap around to negative half-space of vertical axis
                    vert_delta1 = vert_delta1 - size(zf1{i,1},1);
                end
                if horiz_delta1 > size(zf1{i,1},2) / 2  %same for horizontal axis
                    horiz_delta1 = horiz_delta1 - size(zf1{i,1},2);
                end

                pos(i,1:2) = pos(i,1:2) + cell_size * [vert_delta1 - 1, horiz_delta1 - 1];
            end
        
            %obtain a subwindow for training at newly estimated target position
            patch1 = get_subwindow(im, pos(i,1:2), window_sz);
            xf1{i,1,:} = fft2(get_features(patch1, features, cell_size, cos_window));

            %Kernel Ridge Regression, calculate alphas (in Fourier domain)
            switch kernel.type
            case 'gaussian'
                kf1 = gaussian_correlation(xf1{i,1}, xf1{i,1}, kernel.sigma);
            case 'polynomial'
                kf1 = polynomial_correlation(xf1, xf1, kernel.poly_a, kernel.poly_b);
            case 'linear'
                kf1 = linear_correlation(xf1, xf1);
            end
            alphaf1{i,1,:} = yf ./ (kf1 + lambda);   %equation for fast training

            if frame == 1  %first frame, train with a single image
                model_alphaf1{i,1} = alphaf1{i,1};
                model_xf1{i,1} = xf1{i,1};
            else
                %subsequent frames, interpolate model
                model_alphaf1{i,1} = (1 - interp_factor) * model_alphaf1{i,1} + interp_factor * alphaf1{i,1};
                model_xf1{i,1} = (1 - interp_factor) * model_xf1{i,1} + interp_factor * xf1{i,1};
            end
        end

		%visualization
		if show_visualization
			
            for i=1:Size(1,1)
                ppos = pos(i,1:2);
                ttarget_sz = target_sz(i,1:2);
                box(i,1:4) = [ppos([2,1]) - ttarget_sz([2,1])/2, ttarget_sz([2,1])];
            end
            
            mask = detector(im);
            mask = bwmorph(mask, 'close',10);
            mask = bwmorph(mask, 'fill',10);
            bbox=[];
            cents = [];
            ttarget_sz = [];
            cents   = round(blob(mask));
            Size2 = size(cents);
            
            for i=1:Size2(1,1)
                ppos(1,1) = cents(i,2);
                ppos(1,2) = cents(i,1);
                ttarget_sz = target_sz(1,1:2);
                bbox(i,1:4) = [ppos([2,1]) - ttarget_sz([2,1])/2, ttarget_sz([2,1])];
            end
            
            
            if isempty(bbox) == 0
                overlapRatio = bboxOverlapRatio(box,bbox);
                for i = 1:Size(1,1)
                    
                    [match(i,2),match(i,1)] = max(overlapRatio(i,:));
                    %match(i,2) = overlapRatio(i,match(i,1));
                    if match(i,2) > 0
                        
                        box(i,1:4) = bbox(match(i,1), 1:4);
                        
                    end
                end
                
            else
                
                match = [];
                
            end
            matches{frame} = match;
            
            stop1 = update_visualization1(frame, box);
            
			%stop2 = update_visualization2(frame, box2);
			if stop1, break, end  %user pressed Esc, stop early
            drawnow
            
		end
		%F(frame) = getframe(gcf);
        %writeVideo(v,F(frame));
    end
    %close(v);
end