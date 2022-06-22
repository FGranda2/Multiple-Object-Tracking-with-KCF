function [positions, time] = tracker(video_path, img_files, pos, pos2, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)
    %v = VideoWriter('test.avi');
    %open(v);
    %F(numel(img_files)) = struct('cdata',[],'colormap',[]);
    
	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if resize_image
		pos = floor(pos / 2);
        pos2 = floor(pos2 / 2);
		target_sz = floor(target_sz / 2);
	end


	%window size, taking padding into account
	window_sz = floor(target_sz * (1 + padding));   
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
    if isfield(features, 'deep') && features.deep
        yf = fft2(gaussian_shaped_labels(output_sigma, ceil(window_sz / cell_size)));
%         sz = ceil(window_sz/cell_size)-1+4-4;
%         yf = fft2(gaussian_shaped_labels(output_sigma, sz));

    else
        yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
    end

	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
	
	
	if show_visualization  %create video interface
		update_visualization1 = show_video(img_files, video_path, resize_image);
        %update_visualization2 = show_video(img_files, video_path, resize_image);
	end
	

	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision
    
	for frame = 1:numel(img_files)
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
        
		if frame > 1
			%obtain a subwindow for detection at the position from last
			%frame, and convert to Fourier domain (its size is unchanged)
			
            patch1 = get_subwindow(im, pos, window_sz);
			zf1 = fft2(get_features(patch1, features, cell_size, cos_window));
            
            patch2 = get_subwindow(im, pos2, window_sz);
			zf2 = fft2(get_features(patch2, features, cell_size, cos_window));
			
			%calculate response of the classifier at all shifts
			switch kernel.type
			case 'gaussian'
				kzf1 = gaussian_correlation(zf1, model_xf1, kernel.sigma);
                kzf2 = gaussian_correlation(zf2, model_xf2, kernel.sigma);
			case 'polynomial'
				kzf1 = polynomial_correlation(zf1, model_xf1, kernel.poly_a, kernel.poly_b);
                kzf2 = polynomial_correlation(zf2, model_xf2, kernel.poly_a, kernel.poly_b);
			case 'linear'
				kzf1 = linear_correlation(zf1, model_xf1);
                kzf2 = linear_correlation(zf2, model_xf2);
			end
			response1 = real(ifft2(model_alphaf1 .* kzf1));  %equation for fast detection
            response2 = real(ifft2(model_alphaf2 .* kzf2));  %equation for fast detection

			%target location is at the maximum response. we must take into
			%account the fact that, if the target doesn't move, the peak
			%will appear at the top-left corner, not at the center (this is
			%discussed in the paper). the responses wrap around cyclically.
			[vert_delta1, horiz_delta1] = find(response1 == max(response1(:)), 1);
            [vert_delta2, horiz_delta2] = find(response2 == max(response2(:)), 1);
            
                if vert_delta1 > size(zf1,1) / 2  %wrap around to negative half-space of vertical axis
                    vert_delta1 = vert_delta1 - size(zf1,1);
                end
                if horiz_delta1 > size(zf1,2) / 2  %same for horizontal axis
                    horiz_delta1 = horiz_delta1 - size(zf1,2);
                end
                
                if vert_delta2 > size(zf2,1) / 2  %wrap around to negative half-space of vertical axis
                    vert_delta2 = vert_delta2 - size(zf2,1);
                end
                if horiz_delta2 > size(zf2,2) / 2  %same for horizontal axis
                    horiz_delta2 = horiz_delta2 - size(zf2,2);
                end
                
                
			pos = pos + cell_size * [vert_delta1 - 1, horiz_delta1 - 1];
            pos2 = pos2 + cell_size * [vert_delta2 - 1, horiz_delta2 - 1];
		end

		%obtain a subwindow for training at newly estimated target position
		patch1 = get_subwindow(im, pos, window_sz);
        patch2 = get_subwindow(im, pos2, window_sz);
		xf1 = fft2(get_features(patch1, features, cell_size, cos_window));
        xf2 = fft2(get_features(patch2, features, cell_size, cos_window));

		%Kernel Ridge Regression, calculate alphas (in Fourier domain)
		switch kernel.type
		case 'gaussian'
			kf1 = gaussian_correlation(xf1, xf1, kernel.sigma);
            kf2 = gaussian_correlation(xf2, xf2, kernel.sigma);
		case 'polynomial'
			kf1 = polynomial_correlation(xf1, xf1, kernel.poly_a, kernel.poly_b);
            kf2 = polynomial_correlation(xf2, xf2, kernel.poly_a, kernel.poly_b);
		case 'linear'
			kf1 = linear_correlation(xf1, xf1);
            kf2 = linear_correlation(xf2, xf2);
		end
		alphaf1 = yf ./ (kf1 + lambda);   %equation for fast training
        alphaf2 = yf ./ (kf2 + lambda);   %equation for fast training

		if frame == 1  %first frame, train with a single image
			model_alphaf1 = alphaf1;
			model_xf1 = xf1;
            
            model_alphaf2 = alphaf2;
			model_xf2 = xf2;
		else
			%subsequent frames, interpolate model
			model_alphaf1 = (1 - interp_factor) * model_alphaf1 + interp_factor * alphaf1;
			model_xf1 = (1 - interp_factor) * model_xf1 + interp_factor * xf1;
            
            model_alphaf2 = (1 - interp_factor) * model_alphaf2 + interp_factor * alphaf2;
			model_xf2 = (1 - interp_factor) * model_xf2 + interp_factor * xf2;
		end

		%save position and timing
		positions(frame,:) = pos;
		time = time + toc();

		%visualization
		if show_visualization
			
            box1 = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
			box2 = [pos2([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
            stop1 = update_visualization1(frame, box1,box2);
            
			%stop2 = update_visualization2(frame, box2);
			if stop1, break, end  %user pressed Esc, stop early
            
			
            drawnow
            
		end
		%F(frame) = getframe(gcf);
        %writeVideo(v,F(frame));
    end
    %close(v);
	if resize_image
		positions = positions * 2;
	end
end

