% Tower model
% Shlipf, D., et al. (2013)'s method
Mt = 348398.125;                            % Tower mass (kg) 
Mn = 240000;                                % Nacelle mass (kg) 
Mh = 56780;                                 % Hub mass (kg) 
Mb = 17848.770;                             % Blade mass (kg)
Parameters.Mt = 0.25*Mt + Mn + Mh + 3*Mb;   % Modal mass approximation (kg)
% ds = 0.01;                                % Damping ratio - datasheet value (-)
ds = 0.025;                                 % Damping ratio - empirical (-)
f0 = 0.32699;                               % Tower first natural frequency (Hz) 
Parameters.Ct = 4*pi*Parameters.Mt*ds*f0;   % Modal damping approximation (kg/s)
Parameters.Kt = Parameters.Mt*(2*pi*f0)^2;  % Modal stiffness approxmation (kg/s^2)
% Tower state-space model
sys_tower_ss = ss([0 1; -Parameters.Kt/Parameters.Mt -Parameters.Ct/Parameters.Mt],...
                  [0; 1/Parameters.Mt],...
                  [1 0; 0 1; -Parameters.Kt/Parameters.Mt -Parameters.Ct/Parameters.Mt],...
                  [0; 0;1/Parameters.Mt]);
sys_tower_ss_d=c2d(sys_tower_ss,Parameters.Ts,'tustin');
% Tower side-side dynamics
Parameters.sys_tower_ss_d.A=sys_tower_ss_d.A;
Parameters.sys_tower_ss_d.B=sys_tower_ss_d.B;
Parameters.sys_tower_ss_d.C=sys_tower_ss_d.C;
Parameters.sys_tower_ss_d.D=sys_tower_ss_d.D;
% Tower fore-aft dynamics
Parameters.sys_tower_fa_d.A=sys_tower_ss_d.A;
Parameters.sys_tower_fa_d.B=sys_tower_ss_d.B;
Parameters.sys_tower_fa_d.C=sys_tower_ss_d.C;
Parameters.sys_tower_fa_d.D=sys_tower_ss_d.D;