function [x_ulc,y_ulc,w,h] = upperLCorner(x,y,we,he)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

x_ulc = x - (we/2);
y_ulc = y - (he/2);
w = we;
h = he;
end

