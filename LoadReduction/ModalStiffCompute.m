function [ModalStiff1,ModalStiff2] = ModalStiffCompute(ModeShape,TwFAStif,Height,HtFract)
%UNTITLED2 Summary of this function goes here
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

[m2,n2] = size(DeltaCoeff);
for i = 2:m2
    DeltaCoeff2(i-1,:) = (i-1)*DeltaCoeff(i,:);
end

phi2_x1 =0;
phi2_x2 =0;
% Mode Shape 1
for i = 1:m2-1
    phi2_x1 = phi2_x1 + DeltaCoeff2(i,1)*x^(i-1);
end
% Mode Shape 2
for i = 1:m2-1
    phi2_x2 = phi2_x2 + DeltaCoeff2(i,2)*x^(i-1);
end

% Compute the length of each element
N_ele = size(HtFract);

for i = 1:N_ele
    Height_ele(i) = Height*HtFract(i);
end

% Compute the stiff density
for i = 1:N_ele-1
%     EI_x(i) = (TwFAStif(i+1)-TwFAStif(i))/(HtFract(i+1)-HtFract(i))*(x-HtFract(i))+TwFAStif(i);
end

% Modal Mass 1
for i = 1:N_ele-1
    ModalStiff1_ele(i) = TwFAStif(i)*double(int(phi2_x1,x,HtFract(i),HtFract(i+1))) ;
%     StiffMass1_ele(i) = TwFAStif(i)*int(EI_x(i)*phi2_x1,x,HtFract(i),HtFract(i+1)) ;
end
ModalStiff1 = double(sum(ModalStiff1_ele));

% Modal Mass 2
for i = 1:N_ele-1
    ModalStiff2_ele(i) = TwFAStif(i)*int(phi2_x2,x,HtFract(i),HtFract(i+1)) ;
%         StiffMass2_ele(i) = TwFAStif(i)*int(EI_x(i)*phi2_x2,x,HtFract(i),HtFract(i+1)) ;
end
ModalStiff2 = double(sum(ModalStiff2_ele));
end

