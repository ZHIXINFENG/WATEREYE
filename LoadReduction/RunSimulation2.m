clear
clear all
clc

%% Add FAST binaries and input files (*.dat, *.wnd, etc) path
pathFASTv8AD1504 = "./FASTv8_AeroDyn1504"; % Change it to the FAST binary
addpath( pathFASTv8AD1504 );
addpath( genpath('inputfiles') );

%% Load Aerodynamic Coefficient Tables
% 8 m/s
load(strcat('TablesCpCtCmCq[',num2str(8),'ms_option2][JW].mat'),'Tables')
Tables8ms = Tables; clear Tables

% 12 m/s
load(strcat('TablesCpCtCmCq[',num2str(12),'ms_option2][JW].mat'),'Tables')
Tables12ms = Tables; clear Tables

% 16 m/s
load(strcat('TablesCpCtCmCq[',num2str(16),'ms_option2][JW].mat'),'Tables')
Tables16ms = Tables; clear Tables

% 20 m/s
load(strcat('TablesCpCtCmCq[',num2str(20),'ms_option2][JW].mat'),'Tables')
Tables20ms = Tables; clear Tables

% Making the grid
[X,Y,Z] = meshgrid(Tables8ms.TSR, Tables8ms.Pitch, Tables8ms.Azi);
Tables8ms.X = X; Tables12ms.X = X;Tables16ms.X = X;Tables20ms.X = X;
Tables8ms.Y = Y; Tables12ms.Y = Y;Tables16ms.Y = Y;Tables20ms.Y = Y;
Tables8ms.Z = Z; Tables12ms.Z = Z;Tables16ms.Z = Z;Tables20ms.Z = Z;
clear X Y Z

%% MPC Initialization
Init_FAST_EMPC;
load Outlist;
Control.Torque.SpeedA = 4.1*Parameters.G;

%% Wind Speed Setting
% windCase = 'step_wind_8mps_17mps_noshear.wnd';
windCase = 'wind_8mps_uniform_noshear.wnd';
windCasename = windCase(1:find(windCase=='.')-1);
TMax = 160;
if startsWith(windCase,'step')
    TMax = 600;
    if startsWith(windCase,'step_wind_14mps_25mps')
        TMax = 720;
    elseif startsWith(windCase,'step_wind_12mps_14mps')
        TMax = 180;
    elseif startsWith(windCase,'step_wind_14mps_17mps') % including extreme shear case
        TMax = 240;
    end
elseif startsWith(windCase,'turb') || startsWith(windCase,'ETM') % turb: normal turbulence model (NTM); ETM: extreme turbulence model
    TMax = 660;
elseif contains(windCase,'wake') % wake impingement cases
    TMax = 1000;
elseif startsWith(windCase,'wind_10mps_turbulence')
    Tmax = 240;
end

% Write chosen wind case into InflowWind.dat
writewindmodecase(strcat('windcases/',windCase));
FAST= [pwd,filesep 'inputfiles' filesep 'FAST.fst'];

%% Truncate First 60 seconds?
truncateFirst60Seconds = true;
if truncateFirst60Seconds   
    timeStartFAST = 60;
else
    timeStartFAST = 0;
end

A = regexp( fileread('inputfiles\FAST.fst'), '\n', 'split');
A{38} = sprintf(['          ' num2str(timeStartFAST) '   TStart          - Time to begin tabular output (s)']);
fid = fopen('inputfiles\FAST.fst', 'w');
fprintf(fid, '%s\n', A{:});
fclose(fid);