function Cp = DeriveCp(v_wind, omega_r, pitch,t)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    global Parameters Tables
    
    %pitch=0:0.005:20;
%     if t == 0
%         v_wind = 6;
%         omega_r = 41.6471/Parameters.G;
%     end
    
    TSR=omega_r*Parameters.R/v_wind;
    
    if TSR < 2
        TSR = max(2,TSR);
        Cp_derived = max(interp2(Tables.TSR,Tables.Pitch,Tables.Cp,TSR,pitch,'spline'),0);
    else
        TSR = max(2,min(10,TSR));
        Cp_derived = max(interp2(Tables.TSR,Tables.Pitch,Tables.Cp,TSR,pitch,'spline'),0);
    end
    
    Cp = Cp_derived;
end

