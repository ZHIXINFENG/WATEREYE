% clear all; clc; close all
%% adding some toolboxes
% check_dependency
% if ~exist('export_fig', 'file')
%     error('The following file cannot be found: export_fig.m! Please download from https://github.com/altmany/export_fig');
% end
% if ~exist('hline', 'file')
%     error('The following file cannot be found: hline.m! Please download from https://nl.mathworks.com/matlabcentral/fileexchange/1039-hline-and-vline?s_tid=srchtitle');
% end
% if ~exist('vline', 'file')
%     error('The following file cannot be found: vline.m! Please download from https://nl.mathworks.com/matlabcentral/fileexchange/1039-hline-and-vline?s_tid=srchtitle');
% end
% if ~exist('yalmip', 'file')
%     error('YALMIP is not installed! Please download from https://yalmip.github.io/download/');
% end
% if ~exist('mosekdiag', 'file')
%     error('MOSEK is not installed! Please download from https://www.mosek.com/downloads/');
% end

addpath('inputfiles');

global Parameters Solution

isInitialized = false;
plotCurves    = false;

load('NREL5MW.mat');
Control.DT  = 0.01;
RPM_Init    = 9.4965;
P_InitAngle = 0;

%% Set turbine parameters

% load coefficient tables
load('[210405][0023][AD1504DVR]CoefficientTables.mat');

% Ts                    = 0.2;                                      
Parameters.TsMPC        = 0.2;                                      % Sampling time for MPC (s) 
% Solution.Ts=Ts;
Parameters.TsFAST       = 0.01;                                     % Sampling time for FAST (s)
Parameters.Np           = 100;                                      % MPC prediction horizon (-)
Parameters.H            = 90;                                       % Hub-height (m)
Parameters.R            = 63;                                       % Rotor radius (m)
Parameters.A            = pi*Parameters.R^2;                        % Rotor area (m^2)
Parameters.rho          = 1.225;                                    % Air density (kg/m^3)
Parameters.G            = 97;                                       % Gearbox ratio (-)
% Parameters.JHss         = 4.6917e+03;                               % HSS equivalent inertia (kg-m^2)
% Parameters.JLss         = Parameters.JHss*Parameters.G^2;           % LSS equivalent (kg-m^2)
Jr = 35444067;                                                      % LSS equivalent (kg-m^2)
Jg = 534.116;                                                       % HSS equivalent inertia (kg-m^2)0
Parameters.Je = Jg + Jr/(Parameters.G^2);
Parameters.J = Jr + Jg*(Parameters.G^2);


Parameters.Prated       = 5e6;                                      % Rated power (W)
Parameters.Effic        = 0.944;                                    % Gen. efficiency (-)
Parameters.Omegagmax    = 12.1/60*2*pi*Parameters.G*1.3;            % Gen. overspeeding limit (rad/s)
Parameters.Omegagrated  = 12.1/60*2*pi*Parameters.G;                % Gen. rated speed (rad/s)
Parameters.Omegagmin    = 4.1/60*2*pi*Parameters.G;                 % Gen. min. speed (rad/s)  
Parameters.Tmax         = Control.Torque.Demanded;                  % Maximum torque (Nm)
Parameters.ScaleP       = Parameters.Prated;                        % Scaling factor for power (W)
% Parameters.ScaleK       = 1/2*Parameters.JHss*...
%                           Parameters.Omegagmax.^2;                  % Scaling factor for kinetic energy (J)
Parameters.ScaleK = 1/2*Parameters.Je*Parameters.Omegagmax^2;       % Scaling factor for kinetic energy (J)

% Tower model
% Shlipf, D., et al. (2013)'s method
Mt = 348398.125;                            % Tower mass (kg) 
Mn = 240000;                                % Nacelle mass (kg) 
Mh = 56780;                                 % Hub mass (kg) 
Mb = 17848.770;                             % Blade mass (kg))
Parameters.MT = 0.25*Mt + Mn + Mh + 3*Mb;   % Modal mass approximation (kg)
% ds = 0.01;                                % Damping ratio - datasheet value (-)
ds = 0.025;                                 % Damping ratio - empirical (-)
f0 = 0.32699;                               % Tower first natural frequency (Hz) 
Parameters.CT = 4*pi*Parameters.MT*ds*f0;   % Modal damping approximation (kg/s)
Parameters.KT = Parameters.MT*(2*pi*f0)^2;  % Modal stiffness approxmation (kg/s^2)

% Tower state-space model
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
ModeShape = [0 0;...
             0 0;...
             0.7004 -70.5319;...
             2.1963 -63.7623;...
             -5.6202 289.7369;...
             6.2275 -176.5134;...
             -2.504 22.0706];
[ModalMass1,ModalMass2] = ModalMassCompute(ModeShape,TMassDen,Height,HtFract);
Mass1 = 0.25*ModalMass1 + Mn + Mh + 3*Mb;
Mass2 = 0.25*ModalMass2 + Mn + Mh + 3*Mb;
MassMatrix = diag([Mass1 Mass2]);
Mm = MassMatrix;

% Origin Input Matrix
B0 = [1;0];
% Bm = inv(MassMatrix)*Phi'*B0;
											 
											 
%sensitive analysis
%f1 = 0.3140;
%f1 = 0.3340;
f1 = 0.3040;

%f1 = 0.3240;
%f2 = 3.0003;
f2 = 2.9003;
omega1 = 2*pi*f1;
omega2 = 2*pi*f2;
ds1 = 0.01;                                 % Tower first fore-aft damping ratio
ds2 = 0.01;                                  % Tower second fore-aft damping ratio
% d1 = MassMatrix(1,1)*ds1;        % 1st Modal damping approximation (kg/s)
% d2 = MassMatrix(2,2)*ds2;        % 2nd Modal damping approximation (kg/s)
damp1 = 4*pi*f1*ds1;        % 1st Modal damping approximation (kg/s)
damp2 = 4*pi*f2*ds2;        % 2nd Modal damping approximation (kg/s)
B0 = [0 0 0 0 0 0 1]';
Bm = inv(Mm)*ModeShape'*B0;
Parameters.ModeShape = ModeShape;

Am1 = [0 1; -omega1^2 -2*damp1*omega1];          % A matrix
Bm1 = [0;Bm(2)];
Am2 = [0 1; -omega2^2 -2*damp2*omega2];           
Bm2 = [0;Bm(1)];
Cm1 = [1 0;0 1; -omega1^2 -2*damp1*omega1];
Cm2 = [1 0;0 1; -omega2^2 -2*damp2*omega2];
Dm1 = [0; 0; Bm(2)];
Dm2 = [0; 0; Bm(1)];
sys_tower_mode_ss1 = ss (Am1,Bm1,Cm1,Dm1);
sys_tower_mode_d1 = c2d(sys_tower_mode_ss1,Parameters.TsFAST,'tustin');
sys_tower_mode_ss2= ss (Am2,Bm2,Cm2,Dm2);
sys_tower_mode_d2 = c2d(sys_tower_mode_ss2,Parameters.TsFAST,'tustin');

Parameters.sys_tower_fa_d.Am1=sys_tower_mode_d1.A;
Parameters.sys_tower_fa_d.Bm1=sys_tower_mode_d1.B;
Parameters.sys_tower_fa_d.Cm1=sys_tower_mode_d1.C;
Parameters.sys_tower_fa_d.Dm1=sys_tower_mode_d1.D;

Parameters.sys_tower_fa_d.Am2=sys_tower_mode_d2.A;
Parameters.sys_tower_fa_d.Bm2=sys_tower_mode_d2.B;
Parameters.sys_tower_fa_d.Cm2=sys_tower_mode_d2.C;
Parameters.sys_tower_fa_d.Dm2=sys_tower_mode_d2.D;

% Tower Observer Dynamics
% TsObsv = 0.01; 
Parameters.TsObsv = Parameters.TsFAST;


% Position
%Position1 = 1;
Position1 = (9-1/2)*(1/9);
% Position2 = 0.5;
Position2 = (7-1/2)*(1/9);
Position3 = (6-1/2)*(1/9);
Parameters.StaticPosition1 = PositionCompute(Parameters.ModeShape,Position1);
Parameters.StaticPosition2 = PositionCompute(Parameters.ModeShape,Position2);
Parameters.StaticPosition3 = PositionCompute(Parameters.ModeShape,Position3);

% Tower observer state-space model

A_ori = [0 1 0 0; -omega1^2 -2*damp1*omega1 0 0; 0 0 0 1; 0 0 -omega2^2 -2*damp2*omega2];
B_ori = [0; Bm(2); 0; Bm(1)];
C_ori = [-omega1^2 -2*damp1*omega1 0 0; 0 0 -omega2^2 -2*damp2*omega2];
D_ori = [Bm(2);Bm(1)];

TF_Matrix1 = [Parameters.StaticPosition1(1) 0 Parameters.StaticPosition1(2) 0;...
                    0 Parameters.StaticPosition1(1) 0 Parameters.StaticPosition1(2);...
                    Parameters.StaticPosition2(1) 0 Parameters.StaticPosition2(2) 0;...
                    0 Parameters.StaticPosition2(1) 0 Parameters.StaticPosition2(2)];
                
TF_Matrix2 = [Parameters.StaticPosition1(1) Parameters.StaticPosition1(2);...
              Parameters.StaticPosition2(1) Parameters.StaticPosition2(2)];
     

% A_new = TF_Matrix1*A_ori/TF_Matrix1;
% B_new = TF_Matrix1*B_ori;
C_new = TF_Matrix2*C_ori;
D_new = TF_Matrix2*D_ori;

sys_tower_ss_obsv = ss(A_ori,B_ori,C_new,D_new);


% sys_tower_ss_obsv = ss([0 1 0 0; -omega1^2 -2*damp1*omega1 0 0; 0 0 0 1; 0 0 -omega2^2 -2*damp2*omega2],...
%                        [0; Bm(2); 0; Bm(1)],...
%                        [-omega1^2 -2*damp1*omega1 0 0; 0 0 -omega2^2 -2*damp2*omega2],...
%                        [Bm(2);Bm(1)]);    


% sys_tower_ss_obsv1 = ss([0 1; -omega1^2 -2*damp1*omega1],...
%                        [0; Bm(2)],...
%                        [-omega1^2 -2*damp1*omega1],...
%                        [Bm(2)]);
% sys_tower_ss_obsv2 = ss([0 1; -omega2^2 -2*damp2*omega2],...
%                        [0; Bm(1)],...
%                        [-omega2^2 -2*damp2*omega2],...
%                        [Bm(1)]);

sys_tower_fa_d_obsv = c2d(sys_tower_ss_obsv,Parameters.TsObsv,'tustin');

% sys_tower_fa_d_obsv1=c2d(sys_tower_ss_obsv1,Parameters.TsObsv,'tustin');
% sys_tower_fa_d_obsv2=c2d(sys_tower_ss_obsv2,Parameters.TsObsv,'tustin');								   
													   												   

% Tower fore-aft dynamics

Parameters.sys_tower_fa_d_obsv.A=sys_tower_fa_d_obsv.A;
Parameters.sys_tower_fa_d_obsv.B=sys_tower_fa_d_obsv.B;
Parameters.sys_tower_fa_d_obsv.C=sys_tower_fa_d_obsv.C;
Parameters.sys_tower_fa_d_obsv.D=sys_tower_fa_d_obsv.D;


% Parameters.sys_tower_fa_d_obsv1.A=sys_tower_fa_d_obsv1.A;
% Parameters.sys_tower_fa_d_obsv1.B=sys_tower_fa_d_obsv1.B;
% Parameters.sys_tower_fa_d_obsv1.C=sys_tower_fa_d_obsv1.C;
% Parameters.sys_tower_fa_d_obsv1.D=sys_tower_fa_d_obsv1.D;
% 
% Parameters.sys_tower_fa_d_obsv2.A=sys_tower_fa_d_obsv2.A;
% Parameters.sys_tower_fa_d_obsv2.B=sys_tower_fa_d_obsv2.B;
% Parameters.sys_tower_fa_d_obsv2.C=sys_tower_fa_d_obsv2.C;
% Parameters.sys_tower_fa_d_obsv2.D=sys_tower_fa_d_obsv2.D;

%% Compare Observer

sys_tower_ss_obsv_mode = ss(A_ori,B_ori,C_ori,D_ori);
sys_tower_fa_d_obsv_mode = c2d(sys_tower_ss_obsv_mode,Parameters.TsObsv,'tustin');


% A_ori2 = [0 1 0 0; -omega1^2 -2*damp1*omega1 0 0; 0 0 0 1; 0 0 -omega2^2 -2*damp2*omega2];
% B_ori2 = [0; Bm(2); 0; Bm(1)];
% C_ori2 = [-omega1^2 -2*damp1*omega1 0 0; 0 0 -omega2^2 -2*damp2*omega2];
% D_ori2 = [Bm(2);Bm(1)];
% 
% TF_Matrix1_comp = [Parameters.StaticPosition1(1) 0 Parameters.StaticPosition1(2) 0;...
%                     0 Parameters.StaticPosition1(1) 0 Parameters.StaticPosition1(2);...
%                     Parameters.StaticPosition3(1) 0 Parameters.StaticPosition3(2) 0;...
%                     0 Parameters.StaticPosition3(1) 0 Parameters.StaticPosition3(2)];
%                 
% TF_Matrix2_comp = [Parameters.StaticPosition1(1) Parameters.StaticPosition1(2);...
%               Parameters.StaticPosition3(1) Parameters.StaticPosition3(2);];
%      
% 
% % A_new = TF_Matrix1*A_ori/TF_Matrix1;
% % B_new = TF_Matrix1*B_ori;
% C_new_comp = TF_Matrix2_comp*C_ori2;
% D_new_comp = TF_Matrix2_comp*D_ori2;
% 
% sys_tower_ss_obsv_comp = ss(A_ori2,B_ori2,C_new_comp,D_new_comp);
% 
% sys_tower_fa_d_obsv_comp = c2d(sys_tower_ss_obsv_comp,Parameters.TsObsv,'tustin');
% 
% Parameters.sys_tower_fa_d_obsv_comp.A=sys_tower_fa_d_obsv_comp.A;
% Parameters.sys_tower_fa_d_obsv_comp.B=sys_tower_fa_d_obsv_comp.B;
% Parameters.sys_tower_fa_d_obsv_comp.C=sys_tower_fa_d_obsv_comp.C;
% Parameters.sys_tower_fa_d_obsv_comp.D=sys_tower_fa_d_obsv_comp.D;
% 
% luenPolesCont_comp = real(pole(sys_tower_ss_obsv_comp)).*[10; 10.1; 10; 10.1];
% luenGainCont_comp  = place(sys_tower_ss_obsv_comp.A', sys_tower_ss_obsv_comp.C', luenPolesCont_comp )';
% 
% luenPolesDisc_comp = exp(luenPolesCont_comp.*Parameters.TsObsv);
% luenGainDisc_comp  = place(sys_tower_fa_d_obsv_comp.A',sys_tower_fa_d_obsv_comp.C',luenPolesDisc_comp)';

%% Luenberger estimator gain calculation for tower estimation

luenPolesCont = real(pole(sys_tower_ss_obsv)).*[10; 10.1; 10; 10.1];
luenGainCont  = place(sys_tower_ss_obsv.A', sys_tower_ss_obsv.C', luenPolesCont )';

luenPolesDisc = exp(luenPolesCont.*Parameters.TsObsv);
luenGainDisc  = place(sys_tower_fa_d_obsv.A',sys_tower_fa_d_obsv.C',luenPolesDisc)';


% luenPolesCont1 = real(pole(sys_tower_ss_obsv1)).*[10; 10.1];
% luenGainCont1  = place(sys_tower_ss_obsv1.A', sys_tower_ss_obsv1.C', luenPolesCont1 )';
% 
% luenPolesDisc1 = exp(luenPolesCont1.*Parameters.TsObsv);
% luenGainDisc1  = place(sys_tower_fa_d_obsv1.A',sys_tower_fa_d_obsv1.C',luenPolesDisc1)';
% 
% luenPolesCont2 = real(pole(sys_tower_ss_obsv2)).*[10; 10.1];
% luenGainCont2  = place(sys_tower_ss_obsv2.A', sys_tower_ss_obsv2.C', luenPolesCont2 )';
% 
% luenPolesDisc2 = exp(luenPolesCont2.*Parameters.TsObsv);
% luenGainDisc2  = place(sys_tower_fa_d_obsv2.A',sys_tower_fa_d_obsv2.C',luenPolesDisc2)';


%% Create Interpolation Tables

% Reserve full tables
Parameters.Rotor_PitchFull                          = Rotor_Pitch;
Parameters.Rotor_cPFull                             = Rotor_cP_raw_fine;
Parameters.Rotor_cPFull(Parameters.Rotor_cPFull<=0) = 0;
Parameters.Rotor_cQFull                             = Rotor_cQ_raw_fine;
Parameters.Rotor_cTFull                             = Rotor_cT_raw_fine;
Parameters.Rotor_cTFull(Parameters.Rotor_cTFull<=0) = 0;

% Separating Cp,Cq, and Ct values by sign
Rotor_cP_raw_fine_temp                              = Rotor_cP_raw_fine;
Rotor_cP_raw_fine_temp(Rotor_cP_raw_fine_temp>=0)   = 1;
Rotor_cP_raw_fine_temp(Rotor_cP_raw_fine_temp<0)    = 0;

Rotor_cQ_raw_fine_temp                              = Rotor_cQ_raw_fine;
Rotor_cQ_raw_fine_temp(Rotor_cQ_raw_fine_temp>=0)   = 1;
Rotor_cQ_raw_fine_temp(Rotor_cQ_raw_fine_temp<0)    = 0;

Rotor_cT_raw_fine_temp                              = Rotor_cT_raw_fine;
Rotor_cT_raw_fine_temp(Rotor_cT_raw_fine_temp>=0)   = 1;
Rotor_cT_raw_fine_temp(Rotor_cT_raw_fine_temp<0)    = 0;

% Creating meshgrid only containing nonnegative pitch
Parameters.Rotor_Pitch = Rotor_Pitch(Rotor_Pitch>=0);
Parameters.Rotor_Lamda = Rotor_Lamda; 
[Xq,Yq]                = meshgrid(Parameters.Rotor_Lamda, Parameters.Rotor_Pitch);
Parameters.Rotor_cP    = interp2(Rotor_Lamda,Rotor_Pitch,Rotor_cP_raw_fine,Xq,Yq,'linear');
Parameters.Rotor_cQ    = interp2(Rotor_Lamda,Rotor_Pitch,Rotor_cQ_raw_fine,Xq,Yq,'linear');
Parameters.Rotor_cT    = interp2(Rotor_Lamda,Rotor_Pitch,Rotor_cT_raw_fine,Xq,Yq,'linear');
Parameters.Rotor_cTcQ  = Parameters.Rotor_cT./Parameters.Rotor_cQ;

% 'FlyZone' here is defined as the intersection of Cp, Ct, and Cq tables which values are nonnegative with nonnegative pitch angles
FlyZone = Rotor_cP_raw_fine_temp & Rotor_cT_raw_fine_temp & Rotor_cQ_raw_fine_temp;
FlyZone = FlyZone(Rotor_Pitch>=0,:);

[Parameters.dCp_dlambda, Parameters.dCp_dbeta] = gradient( Parameters.Rotor_cP, mean( diff( Xq(1,:) ) ), mean( diff( Yq(:,1) ) ) );
[Parameters.dCq_dlambda, Parameters.dCq_dbeta] = gradient( Parameters.Rotor_cQ, mean( diff( Xq(1,:) ) ), mean( diff( Yq(:,1) ) ) );
[Parameters.dCt_dlambda, Parameters.dCt_dbeta] = gradient( Parameters.Rotor_cT, mean( diff( Xq(1,:) ) ), mean( diff( Yq(:,1) ) ) );

% Clean up Cp table which has positive partial derivative dCp/dPitch (i.e. corresponding to stall pitch angles)
Parameters.Rotor_cP(Parameters.dCp_dbeta>0) = 0;

% Preallocation for clean Cp, Ct, and Cq tables
zerosTable = zeros(length(Parameters.Rotor_Pitch),length(Parameters.Rotor_Lamda));

Rotor_cP_clean                              = zerosTable;
Rotor_cP_clean(FlyZone)                     = Parameters.Rotor_cP(FlyZone);
Parameters.Rotor_cP                         = Rotor_cP_clean;
Parameters.Rotor_cP(Parameters.Rotor_cP==0) = NaN;

Rotor_cT_clean          = zerosTable;
Rotor_cT_clean(FlyZone) = Parameters.Rotor_cT(FlyZone);
Parameters.Rotor_cT     = Rotor_cT_clean;

Rotor_cQ_clean          = zerosTable;
Rotor_cQ_clean(FlyZone) = Parameters.Rotor_cQ(FlyZone);
Parameters.Rotor_cQ     = Rotor_cQ_clean;

Rotor_cTcQ_clean          = zerosTable;
Rotor_cTcQ_clean(FlyZone) = Parameters.Rotor_cTcQ(FlyZone);
Parameters.Rotor_cTcQ     = Rotor_cTcQ_clean;

%% Compute optimal mode gain for Torque=k*omega^2

[II,JJ]=max(Parameters.Rotor_cP);
[maxCp,III]=max(II); Pitch_opt=Yq(JJ(III),III); TSR_opt=Xq(JJ(III),III);
Kopt=1/2*Parameters.rho*pi*Parameters.R^5*maxCp/Parameters.G^3/TSR_opt^3*Parameters.Effic;

%% Calculating Available Power PWL Coefficients for Aerodynamic Power Upper Bound
PWA_partitions=[0.05:0.05:0.6 0.35:0.05:0.75 0.8:0.02:1.2 1.25:0.25:1.55 2.5:0.5:5]';
Parameters.Nconst=length(PWA_partitions); % number of lines

counter = 0;
for V = 2:0.25:25
    counter = counter+1;

%     Rotor_K = 1/2*(Rotor_Lamda*V/Parameters.R*Parameters.G).^2*Parameters.JHss./Parameters.ScaleK;
    Rotor_K = 1/2*(Rotor_Lamda*V/Parameters.R*Parameters.G).^2*Parameters.Je./Parameters.ScaleK;

%     Rotor_Kls_max = min( max(Rotor_K)*0.7, 1/2*Parameters.JHss*Parameters.Omegagmax.^2*10 );
    Rotor_Kls_max = min( max(Rotor_K)*0.7, 1/2*Parameters.Je*Parameters.Omegagmax.^2*10 );
    Rotor_Kls     = 0:(Rotor_Kls_max/2000):Rotor_Kls_max;

    [Xq,Yq] = meshgrid(Rotor_Kls, 0:0.025:25);
    Cpp = max(interp2(Rotor_K,Rotor_Pitch(Rotor_Pitch>=0),Rotor_cP_clean,Xq,Yq,'linear'));

    Paero_normal = 1/2*Parameters.rho*pi*Parameters.R^2*Cpp/Parameters.ScaleP;
    Paero        = Paero_normal*V^3;

    if plotCurves
        ff=figure(2);ff.Units='centimeters'; ff.Position=[2 2 48 24]; ff.Color='white'; 
        plot(Rotor_Kls,Paero_normal); hold on
    end

    [~,idxPaeroNorm]=max(Paero_normal);

    Rotor_K_partitions      = Rotor_Kls(idxPaeroNorm).*PWA_partitions;
    Paero_normal_partitions = interp1(Rotor_Kls,Paero_normal,Rotor_K_partitions,'pchip');

    Bc = interp1(Rotor_Kls,gradient(Paero_normal)./mean(diff(Rotor_Kls)),Rotor_K_partitions,'pchip');
    Ac = Paero_normal_partitions-Bc(:).*Rotor_K_partitions;

    if plotCurves
        plot(Rotor_K_partitions,Paero_normal_partitions,'+');
    end
    for i=1:length(Rotor_K_partitions)
        if plotCurves
            plot(Rotor_Kls,Ac(i)+Bc(i)*Rotor_Kls,'k:')
        end
        Complete_set(:,i)=Ac(i)+Bc(i)*Rotor_Kls;
    end
    if plotCurves
        plot(Rotor_Kls,min(squeeze(Complete_set)')','k','Linewidth',2)
    end
    minCompleteSet = min(squeeze(Complete_set)');
    NORM = vaf(minCompleteSet(minCompleteSet>0),Paero_normal(minCompleteSet>0));

    [maxNorm,idxMaxNorm]=max(NORM); 
    %display(num2str(maxNorm));
    Parameters.BOUNDPW.V(counter)=V;
    Parameters.BOUNDPW.Bc(:,counter)=Bc(:,idxMaxNorm);
    Parameters.BOUNDPW.Ac(:,counter)=Ac(:,idxMaxNorm);
    clear Complete_set;
%     if plotCurves
%         plot(Rotor_Kls,sqrt(2*Rotor_Kls*Parameters.ScaleK/Parameters.JHss)*Parameters.Tmax/Parameters.ScaleP/V^3,'r','linewidth',3)
%     end
end
save Parameters 'Parameters'

isInitialized = true;