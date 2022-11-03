function WTout = DriveTrain(Cp,v,omega_g_in,t)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    global Parameters

if t == 0
    omega_g_in = 41.6471;
end
    
    omega_r_in = omega_g_in/Parameters.G;

    Tg = 2.3323*omega_g_in^2;
    Pr = 1/2*Parameters.rho*pi*Parameters.R^2*Cp*v^3;
    Tr = Pr/omega_r_in;
    
    omega_g = Parameters.Ts*(Tr/Parameters.G-Tg)/Parameters.Je+omega_g_in;
    omega_r = omega_g/Parameters.G;

    Pg = Tg*omega_g;

    WTout = [omega_r;omega_g;Pr;Tr;Pg;Tg];

end

