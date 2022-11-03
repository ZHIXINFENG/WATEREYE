function [MassMatrix] = MassCompute(HtFract,Density,Height)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
N = size(HtFract);

for i = 2:N
        MassEle(i-1) = (HtFract(i)-HtFract(i-1)) * Height *Density(i-1);
end


% MassDiag(1) = sum(MassEle(1:2));
% MassDiag(2) = sum(MassEle(3:5));

MassMatrix = diag(MassEle);
end

