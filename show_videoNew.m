function update_visualization_func = show_videoNew(img_files, video_path, resize_image, numBox)
%SHOW_VIDEO
%   Visualizes a tracker in an interactive figure, given a cell array of
%   image file names, their path, and whether to resize the images to
%   half size or not.
%
%   This function returns an UPDATE_VISUALIZATION function handle, that
%   can be called with a frame number and a bounding box [x, y, width,
%   height], as soon as the results for a new frame have been calculated.
%   This way, your results are shown in real-time, but they are also
%   remembered so you can navigate and inspect the video afterwards.
%   Press 'Esc' to send a stop signal (returned by UPDATE_VISUALIZATION).
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


	%store one instance per frame
	num_frames = numel(img_files);
	boxes1 = cell(num_frames,numBox);

	%create window
	[fig_h, axes_h, unused, scroll] = videofig(num_frames, @redraw, [], [], @on_key_press);  %#ok, unused outputs
	%set(fig_h, 'Number','off', 'Name', ['Tracker - ' video_path])
    set(fig_h, 'Name', ['Tracker - ' video_path])
	axis off;
	
	%image and rectangle handles start empty, they are initialized later
	im_h = [];
	rect_h1 = cell(1,numBox);
    for j = 1:numBox
       
        rect_h1{1,j} = [];
        
    end
	
	update_visualization_func = @update_visualization;
	stop_tracker = false;
	

	function stop = update_visualization(frame, box1, newNumber)
        stop = numBox;
        if newNumber ~=numBox
            stop = newNumber;
        end
        
        for i=1:stop
            %store the tracker instance for one frame, and show it. returns
            %true if processing should stop (user pressed 'Esc').
            boxes1{frame,i} = box1(i,1:4);
        end
        scroll(frame);
        stop = stop_tracker;
	end

	function redraw(frame)
		%render main image
		im = imread([video_path img_files{frame}]);
		if size(im,3) > 1,
			im = rgb2gray(im);
		end
		if resize_image,
			im = imresize(im, 0.5);
		end
		
		if isempty(im_h),  %create image
			im_h = imshow(im, 'Border','tight', 'InitialMag',200, 'Parent',axes_h);
		else  %just update it
			set(im_h, 'CData', im)
        end
		
        
        for k=1:numBox
            %render target bounding box for this frame
            if isempty(rect_h1{1,k}) && frame == 1 %create it for the first time
                rect_h1{1,k} = rectangle('Position',[0,0,1,1], 'EdgeColor','g', 'Parent',axes_h);
            end
            if ~isempty(boxes1{frame,k}),
                set(rect_h1{1,k}, 'Visible', 'on', 'Position', boxes1{frame,k});
            else
                set(rect_h1{1,k}, 'Visible', 'off');
            end
        end
	end

	function on_key_press(key)
		if strcmp(key, 'escape'),  %stop on 'Esc'
			stop_tracker = true;
		end
	end

end

