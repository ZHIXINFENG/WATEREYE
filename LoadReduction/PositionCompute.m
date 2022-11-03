function [StaticPosition] = PositionCompute(ModeShape,Position)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[m,n] = size(ModeShape);
for i=1:m
    StaticPosition_ele(i,1) = ModeShape(i,1)*Position^(i-1);
    StaticPosition_ele(i,2) = ModeShape(i,2)*Position^(i-1);
end

StaticPosition(1,1) = sum(StaticPosition_ele(:,1));
StaticPosition(1,2) = sum(StaticPosition_ele(:,2));
end

