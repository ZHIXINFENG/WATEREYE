% Parameters Initialization
clear
clc
clear global
global Parameters Tables Ts
load('Parameters.mat');
load('TablesCpCtCq.mat');
Jr = 35444067;
Jg = 534.116;
Parameters.Je = Jg + Jr/(Parameters.G^2);
Parameters.J = Jr + Jg*(Parameters.G^2);
Parameters.ScaleK = 1/2*Parameters.Je*Parameters.Omegagmax^2;
% Tower Vibration Parameters
Mt = 348398.125;                            % Tower mass (kg) 
Mn = 240000;                                % Nacelle mass (kg) 
Mh = 56780;                                 % Hub mass (kg) 
Mb = 17848.770;                             % Blade mass (kg)
Parameters.MT = 0.25*Mt + Mn + Mh + 3*Mb;   % Modal mass approximation (kg)
% ds = 0.01;                                % Damping ratio - datasheet value (-)
ds = 0.025;                                 % Damping ratio - empirical (-)
f0 = 0.32699;                               % Tower first natural frequency (Hz)
Parameters.CT = 4*pi*Parameters.MT*ds*f0;   % Modal damping approximation (kg/s)
Parameters.KT = Parameters.MT*(2*pi*f0)^2;  % Modal stiffness approxmation (kg/s^2)
Parameters.Ts = 0.01;
Ts = Parameters.Ts;
Parameters.Np = 1000;

Height = 87.6;
HtFract = 0:0.1:1;
HtFract = HtFract';
TMassDen = [5590.87;...
            5232.43;...
            4885.76;...
            4550.87;...
            4227.75;...
            3916.41;...
            3616.83;...
            3329.03;...
            3053.01;...
            2788.75;...
            2536.27];
TwFAStif = [614.34*10^9;...
            534.82*10^9;...
            463.27*10^9;...
            399.13*10^9;...
            341.88*10^9;...
            291.01*10^9;...
            246.03*10^9;...
            206.46*10^9;...
            171.85*10^9;...
            141.78*10^9;...
            115.82*10^9];
TwSSStif = [614.34*10^9;...
            534.82*10^9;...
            463.27*10^9;...
            399.13*10^9;...
            341.88*10^9;...
            291.01*10^9;...
            246.03*10^9;...
            206.46*10^9;...
            171.85*10^9;...
            141.78*10^9;...
            115.82*10^9];
% ModeShape = [0 0;...
%              0 0;...
%              0.7004 -70.5319;...
%              2.1963 -63.7623;...
%              -5.6202 289.7369;...
%              6.2275 -176.5134;...
%              -2.504 22.0706];
ModeShape = [0 0;...
             0 0;...
             1.0217 -32.7087;...
             0.1404 12.3847;...
             -0.1612 45.5259;...
             0.0904 -16.4375;...
             -0.0913 7.7644];
[ModalMass1,ModalMass2] = ModalMassCompute(ModeShape,TMassDen,Height,HtFract);
Mass1 = ModalMass1 + Mn + Mh + 3*Mb;
Mass2 = ModalMass2 + Mn + Mh + 3*Mb;
Mm = diag([Mass1 Mass2]);


% Origin Input Matrix
% B0 = [1;0];
% Bm = inv(MassMatrix)*Phi'*B0;

f1 = 0.3240;
f2 = 2.9003;
omega1 = 2*pi*f1;
omega2 = 2*pi*f2;
ds1 = 0.01;                                 % Tower first fore-aft damping ratio
ds2 = 0.01;                                  % Tower second fore-aft damping ratio
% d1 = MassMatrix(1,1)*ds1;        % 1st Modal damping approximation (kg/s)
% d2 = MassMatrix(2,2)*ds2;        % 2nd Modal damping approximation (kg/s)
damp1 = 4*pi*f1*ds1;        % 1st Modal damping approximation (kg/s)
damp2 = 4*pi*f2*ds2;        % 2nd Modal damping approximation (kg/s)

% StiffMatrix = [k1+k2 -k2;-k2 k2];
% StiffMatrix = MassMatrix*(OmegaMatrix^2);
% DampMatrix = [d1+d2 -d2;-d2 d2];
% [ModeShape_Init,OmegaMatrix_square] = eig(inv(MassMatrix)*StiffMatrix);
% OmegaMatrix_Init = OmegaMatrix_square^0.5;
% [OmegaMatrix_sort,Omega_index] = sort(diag(OmegaMatrix_Init),'ascend');
% OmegaMatrix = diag(OmegaMatrix_sort);
% ModeShape = ModeShape_Init(:,Omega_index);
% Mm = ModeShape'*MassMatrix*ModeShape;
% Km = ModeShape'*StiffMatrix*ModeShape;
% Dm = ModeShape'*DampMatrix*ModeShape;
% Z = 0.5*inv(Mm)*Dm*inv(OmegaMatrix);
% damp1 = Z(1,1);
% damp2 = Z(2,2);
% omega1 = OmegaMatrix(1);
% omega2 = OmegaMatrix(2);
% B0 = [0 0 0 0 0 0 1]';
% Bm = inv(Mm)*ModeShape'*B0;
B0 = [0 0 0 0 0 0 0 0 0 1]';
for i = 1:10
    MS_new(i,:) = PositionCompute(ModeShape,0.1*i)';
%     MS_new(i,2) = PositionComupte(ModeShape,0.1*i);
end
Bm = inv(Mm)*MS_new'*B0;
Parameters.MS = ModeShape;

% %% Test
% A_test = [0 1; -omega1^2 -2*damp1*omega1];          % A matrix
% B_test = [0;1/Parameters.MT];
% C_test = [1 0;0 1; -omega1^2 -2*damp1*omega1];
% D_test = [0;0;1/Parameters.MT];
% sys_tower_mode_ss_test = ss(A_test,B_test,C_test,D_test);
% sys_tower_mode_d_test = c2d(sys_tower_mode_ss_test,Parameters.Ts,'tustin');
% 
% Parameters.sys_tower_fa_d_test.A=sys_tower_mode_d_test.A;
% Parameters.sys_tower_fa_d_test.B=sys_tower_mode_d_test.B;
% Parameters.sys_tower_fa_d_test.C=sys_tower_mode_d_test.C;
% Parameters.sys_tower_fa_d_test.D=sys_tower_mode_d_test.D;



% Modal model
Am1 = [0 1; -omega1^2 -2*damp1*omega1];          % A matrix
Bm1 = [0;Bm(1)];
Am2 = [0 1; -omega2^2 -2*damp2*omega2];           
Bm2 = [0;Bm(2)];
Cm1 = [1 0;0 1; -omega1^2 -2*damp1*omega1];
Cm2 = [1 0;0 1; -omega2^2 -2*damp2*omega2];
% Dm1 = [0; 0; Bm(1)];
% Dm2 = [0; 0; Bm(2)];
Dm1 = [0; 0; 0];
Dm2 = [0; 0; 0];
sys_tower_mode_ss1 = ss (Am1,Bm1,Cm1,Dm1);
sys_tower_mode_d1 = c2d(sys_tower_mode_ss1,Parameters.Ts,'tustin');
sys_tower_mode_ss2= ss (Am2,Bm2,Cm2,Dm2);
sys_tower_mode_d2 = c2d(sys_tower_mode_ss2,Parameters.Ts,'tustin');

Parameters.sys_tower_fa_d.Am1=sys_tower_mode_d1.A;
Parameters.sys_tower_fa_d.Bm1=sys_tower_mode_d1.B;
Parameters.sys_tower_fa_d.Cm1=sys_tower_mode_d1.C;
Parameters.sys_tower_fa_d.Dm1=sys_tower_mode_d1.D;

Parameters.sys_tower_fa_d.Am2=sys_tower_mode_d2.A;
Parameters.sys_tower_fa_d.Bm2=sys_tower_mode_d2.B;
Parameters.sys_tower_fa_d.Cm2=sys_tower_mode_d2.C;
Parameters.sys_tower_fa_d.Dm2=sys_tower_mode_d2.D;

P1 = 1;
% Position2 = 0.5;
P2 = 0.72;
Parameters.Tmatrix(1,:) = PositionCompute(Parameters.MS,P1);
Parameters.Tmatrix(2,:) = PositionCompute(Parameters.MS,P2);
Parameters.tTm = [Parameters.Tmatrix(1,1) 0 0 Parameters.Tmatrix(1,2) 0 0;...
                  0 Parameters.Tmatrix(1,1) 0 0 Parameters.Tmatrix(1,2) 0;...
                  0 0 Parameters.Tmatrix(1,1) 0 0 Parameters.Tmatrix(1,2);...
                  Parameters.Tmatrix(2,1) 0 0 Parameters.Tmatrix(2,2) 0 0;...
                  0 Parameters.Tmatrix(2,1) 0 0 Parameters.Tmatrix(2,2) 0;...
                  0 0 Parameters.Tmatrix(2,1) 0 0 Parameters.Tmatrix(2,2)];

A = blkdiag(Am1,Am2);
B = [Bm1;Bm2];
C = Parameters.tTm*blkdiag(Cm1,Cm2);
D = Parameters.tTm*[Dm1;Dm2];

Parameters.tTm_xv = [   Parameters.Tmatrix(1,1) 0 Parameters.Tmatrix(1,2) 0;...
                        0 Parameters.Tmatrix(1,1) 0 Parameters.Tmatrix(1,2);...
                        Parameters.Tmatrix(2,1) 0 Parameters.Tmatrix(2,2) 0;...
                        0 Parameters.Tmatrix(2,1) 0 Parameters.Tmatrix(2,2)];

sys_tower_mode_ss= ss (A,B,C,D);
sys_tower_mode_d = c2d(sys_tower_mode_ss,Parameters.Ts,'tustin');

Parameters.sys_tower_fa_d.A=sys_tower_mode_d.A;
Parameters.sys_tower_fa_d.B=sys_tower_mode_d.B;
Parameters.sys_tower_fa_d.C=sys_tower_mode_d.C;
Parameters.sys_tower_fa_d.D=sys_tower_mode_d.D;

% sys_tower_ss = ss([0 1; -Parameters.KT/Parameters.MT -Parameters.CT/Parameters.MT],...
%                   [0; 1/Parameters.MT],...
%                   [1 0; 0 1; -Parameters.KT/Parameters.MT -Parameters.CT/Parameters.MT],...
%                   [0; 0;1/Parameters.MT]);
% % sys_tower_ss_d=c2d(sys_tower_ss,Parameters.Ts,'tustin');
% sys_tower_ss_d=c2d(sys_tower_ss,Parameters.Ts,'zoh');
% % Tower fore-aft dynamics
% Parameters.sys_tower_fa_d.Am1=sys_tower_ss_d.A;
% Parameters.sys_tower_fa_d.Bm1=sys_tower_ss_d.B;
% Parameters.sys_tower_fa_d.Cm1=sys_tower_ss_d.C;
% Parameters.sys_tower_fa_d.Dm1=sys_tower_ss_d.D;

[Ainit,Binit, fraction, idxMax] = DeriveAB(6);
Parameters.Ainit = Ainit;
Parameters.Binit = Binit;

load('WindSpeed.mat')

% time = 0:Parameters.Ts:12*100;
% time = 0:Parameters.Ts:3*100;
% time = 0:Parameters.Ts:1*100;
% i=1;
% for i = 1:24-10+1
% for i=1:17-6+1
%     WS((i-1)*(100/Parameters.Ts)+1:i*(100/Parameters.Ts)) = (i+6-1)*ones(100/Parameters.Ts,1);
% end
% WSpeed(1) = 6;
% N_WS = size(WS');
% WSpeed(2:N_WS+1) = WS;
% WindSpeed.time = time';
% WindSpeed.signals.values = WSpeed';
% WindSpeed.signals.dimensions = 1;

time = 0:Parameters.Ts:11*100;
for i=1:11
    WS((i-1)*(100/Parameters.Ts)+1:i*(100/Parameters.Ts)) = (6+(i-1)*1.5)*ones(100/Parameters.Ts,1);
end
WSpeed(1) = 6;
N_WS = size(WS');
WSpeed(2:N_WS+1) = WS;
WindSpeed.time = time';
WindSpeed.signals.values = WSpeed';
WindSpeed.signals.dimensions = 1;


%% Genreate Wind Speed
% time = 0:Parameters.Ts:15*100;
% for i = 1:24-10+1
%     WS((i-1)*(100/Parameters.Ts)+1:i*(100/Parameters.Ts)) = (i+8-1)*ones(100/Parameters.Ts,1);
% end
% WSpeed(1) = 8;
% N_WS = size(WS');
% WSpeed(2:N_WS+1) = WS;
% WindSpeed.time = time';
% WindSpeed.signals.values = WSpeed';
% WindSpeed.signals.dimensions = 1;


