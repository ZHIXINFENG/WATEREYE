function [ModalMass1,ModalMass2] = ModalMassCompute(ModeShape,TMassDen,Height,HtFract)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Compute the deferiential coefficient
[m,n] = size(ModeShape);
for i = 2:m
    DeltaCoeff(i-1,:) = (i-1)*ModeShape(i,:);
end

syms x
phi_x1 = 0;
phi_x2 = 0;
% Mode Shape 1
for i = 1:m-1
    phi_x1 = phi_x1 + DeltaCoeff(i,1)*x^(i-1);
end
% Mode Shape 2
for i = 1:m-1
    phi_x2 = phi_x2 + DeltaCoeff(i,2)*x^(i-1);
end

% Compute the length of each element
N_ele = size(HtFract);

for i = 1:N_ele
    Height_ele(i) = Height*HtFract(i);
end

% Compute the modal mass

% Modal Mass 1
for i = 1:N_ele-1
    ModalMass1_ele(i) = (TMassDen(i)+TMassDen(i+1))/2*int(phi_x1^2,x,HtFract(i),HtFract(i+1)) ;
end
ModalMass1 = double(sum(ModalMass1_ele));

% Modal Mass 2
for i = 1:N_ele-1
    ModalMass2_ele(i) = (TMassDen(i)+TMassDen(i+1))/2*int(phi_x2^2,x,HtFract(i),HtFract(i+1)) ;
end
ModalMass2 = double(sum(ModalMass2_ele));

end

