function [A,B, fraction, idxMax] = DeriveAB(currentV)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% Prevent the wind speed fall outside 2-25 m/s range where the PWL
% functions are defined
global Parameters
currentV=max(min(currentV,25),2);

% Preallocation
A=zeros(Parameters.Nconst,1);
B=zeros(Parameters.Nconst,1);

% Determine the fraction for PWL function coefficient calculation
[~,idxMax]=max( ( (Parameters.BOUNDPW.V-currentV) >=0 ) * -50 + (Parameters.BOUNDPW.V-currentV) );
fraction=1-(currentV-Parameters.BOUNDPW.V(idxMax))/(Parameters.BOUNDPW.V(idxMax+1)-Parameters.BOUNDPW.V(idxMax));

% Calculate the PWL function coefficients
A=(Parameters.BOUNDPW.Ac(:,idxMax)*fraction+Parameters.BOUNDPW.Ac(:,idxMax+1)*(1-fraction));
B=(Parameters.BOUNDPW.Bc(:,idxMax)*fraction+Parameters.BOUNDPW.Bc(:,idxMax+1)*(1-fraction));
end

