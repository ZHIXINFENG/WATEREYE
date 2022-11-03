function uuout = DeriveFT(beta,omega_g,v,t)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
global Parameters

if t == 0
    omega_g = 41.6471;
    v = 6;
end

omega_r = omega_g/Parameters.G;
TSR = omega_r*Parameters.R/v;

if TSR <2 
    ct = 0;
else
    TSR = max(2,min(10,TSR));
    ct = interp2(Parameters.Rotor_Lamda,Parameters.Rotor_Pitch,Parameters.Rotor_cT,TSR,beta,'spline');
end

FT = 1/2*Parameters.rho*pi*Parameters.R^2*ct*v^3;

uuout = [FT;ct];
end

