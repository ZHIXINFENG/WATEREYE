function towerout = TowerDynamic(FT,currentX,currentV)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
global Parameters
% Xout = currentV*Parameters.Ts + currentX;
% Vout = 1/Parameters.MT*Parameters.Ts*(FT-Parameters.BT*currentV-Parameters.KT*currentX)+currentV;
% Aout = 1/Parameters.MT*(FT-Parameters.BT*currentV-Parameters.KT*currentX);
towerstates = Parameters.sys_tower_fa_d.A*[currentX;currentV]+ Parameters.sys_tower_fa_d.B*FT;
towerout = Parameters.sys_tower_fa_d.C*towerstates+ Parameters.sys_tower_fa_d.D*FT;
% towerout = [Xout;Vout;Aout];
end

