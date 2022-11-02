% This is the version of code that emplement the forecast in a sequence of time

%% Preprocessing
clear
clc
load('Temperature.mat');
T_1 = Temp(1,:);    % Becasue we have the data at 0, 6, 12, 18, here we use the data at 0 hour as everyday's measurement

%% Preprocessing 2
T = Temp(1,:);
T = T(1,19:4890)-273.15;
T = T';
Nyear = floor(size(T)/365);

%% GPR Fit Each Year
Nyear = floor(size(T)/365);
for i = 1:Nyear
    Trs(:,i) = T(((i-1)*365+1):i*365);
end

Tos = T(i*365+1:size(T));
%%
Days = 1:365*10;
Days = Days';

T_series = [];
for i = 1:10
    T_series = [T_series;Trs(:,i)];
end

% This part is the original version, basis function H=1
    %Days_prd = 1:365*3;
    %Days_prd = Days_prd';
    %TempMd = fitrgp(Days,T_series,'KernelFunction','exponential');
    %[ypred,~,yint] = predict(TempMd,Days_prd,'Alpha',0.1);

% This part is the original version, basis function H=sin(2*pi/364*x)+x
D = size(Days,2);
%TempMd = fitrgp(Trs(1:size(Tos),:),Tos);
%TempMd_months = fitrgp(Months,T_series_month,'KernelFunction','exponential');
%[ypred_month,~,yint_month] = predict(TempMd_months,Months_prd,'Alpha',0.1);
%[ypred_month,~,yint_month] = predict(TempMd_months,Months_prd,'Alpha',0.1);
%ypred = forecast(TempMd,T_series,Days_prd);
sigmaL10 = 0.1*ones(D(:,1),1);
sigmaL20 = 0.1;
sigmaF10 = 1;
sigmaF20 = 1;        
theta0   = [log(sigmaL10);log(sigmaL20);log(sigmaF10);log(sigmaF20)];

Days_prd = 365*10+1:365*13;
Days_prd = Days_prd';

hfcn = @(X)[X,1.5*sin(2*pi/364*X)];
beta0 = [1;1];
%TempMd = fitrgp(Days,T_series,'Basis',hfcn,'beta',beta0,'KernelFunction','exponential','Sigma',2,'ComputationMethod','v');
TempMd_custom = fitrgp(Days,T_series,'kernelfunction',@mykernal,'kernelparameters',theta0,'Basis',hfcn,'beta',beta0);
[ypred_custom,~,yint_custom] = predict(TempMd_custom,Days_prd,'Alpha',0.1);

%New Plot for GPR by day

subplot(2,1,1)
%T_new = ypred(365*10+1:365*13);
T_new_custom = ypred_custom;
T_his_custom = [Trs(:,11);Trs(:,12);Trs(:,13)];  % History data
X = 1:365*3;
X = X';
scatter(X,T_new_custom,'MarkerEdgeColor','#0072BD')
hold on
scatter(X,T_his_custom,'MarkerEdgeColor','#D95319')
patch([X;flipud(X)],[yint_custom(:,1);flipud(yint_custom(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
%patch([X;flipud(X)],[yint(365*10+1:365*13,1);flipud(yint(365*10+1:365*13,2))],'k','FaceAlpha',0.1); % Prediction intervals
xlim([0 365*3])
legend('Prediction','Real Data','Intervel','Location','southeast')
xlabel('Days')
ylabel('Temperature/K')
title('Prediction Result per day')


subplot(2,1,2)
scatter(X,T_new_custom-T_his_custom);
xlim([0 365*3])
xlabel('Days')
ylabel('Error of Temperature/K')
title('Error of prediction per month')

%% Fit GPR with 10 years history and 3 years prediction by month
clc

Months = 1:12*10;
Months = Months';
N_month = 12;
% Generate the monthly average data
for i = 1:13
    for j = 1:N_month
        Trs_month(j,i) = sum(Trs((j-1)*30+1:j*30,i))/30;
    end
end

T_series_month = [];
for i = 1:10
    T_series_month = [T_series_month;Trs_month(:,i)];
end

%Months_prd = 1:12*3;
%Months_prd = Months_prd';
% Days2 = 365+1:365*2;
% Days2 = Days2';
D = size(Months,2);
%TempMd = fitrgp(Trs(1:size(Tos),:),Tos);
%TempMd_months = fitrgp(Months,T_series_month,'KernelFunction','exponential');
%[ypred_month,~,yint_month] = predict(TempMd_months,Months_prd,'Alpha',0.1);
%[ypred_month,~,yint_month] = predict(TempMd_months,Months_prd,'Alpha',0.1);
%ypred = forecast(TempMd,T_series,Days_prd);
sigmaL10 = 0.1*ones(D(:,1),1);
sigmaL20 = 0.1;
sigmaF10 = 1;
sigmaF20 = 1;        
theta0   = [log(sigmaL10);log(sigmaL20);log(sigmaF10);log(sigmaF20)];
% H function x+sinx
Months_prd = 12*10+1:12*13;
Months_prd = Months_prd';
hfcn = @(X)[X, sin(2*pi/12*X)];
beta0 = [1;1];
%TempMd_months = fitrgp(Months,T_series_month,'KernelFunction','ardrationalquadratic','Basis',hfcn,'beta',beta0,'Sigma',2,'ComputationMethod','v','Optimizer','fminsearch');
%TempMd_months = fitrgp(Months,T_series_month,'Basis',hfcn,'beta',beta0);
TempMd_months_custom = fitrgp(Months,T_series_month,'kernelfunction',@mykernal,'kernelparameters',theta0,'Basis',hfcn,'beta',beta0);
[ypred_month_custom,~,yint_month_custom] = predict(TempMd_months_custom,Months_prd,'Alpha',0.1);

%Trs_t = Trs((size(Tos)+1:365),:);
%[ypred1,~,yint1] = predict(TempMd,Trs_t);

%New Plot for GPR by day
subplot(2,1,1)
T_new_months_custom = ypred_month_custom;
T_his_months_custom = [Trs_month(:,11);Trs_month(:,12);Trs_month(:,13)];
% X = 1:12*3;
% X = X';
X_months = datetime(2018,1,1) + calmonths(0:12*3-1);
X_months = X_months';
scatter(X_months,T_new_months_custom,'MarkerEdgeColor','#0072BD')
hold on
scatter(X_months,T_his_months_custom,'MarkerEdgeColor','#D95319')
patch([X_months;flipud(X_months)],[yint_month_custom(:,1);flipud(yint_month_custom(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
legend('Prediction','Real Data','90% Confidence Intervel','Location','southeast')
xlabel('Months')
ylabel('Temperature/K')
title('Prediction Result per Month using Custom Kernel')
%xlim([0 36])
hold off

subplot(2,1,2)
scatter(X_months,(T_new_months_custom-T_his_months_custom)./T_his_months_custom);
%5xlim([0 36])
xlabel('Months')
ylabel('Relative Error of Temperature')
title('Relative Error of Prediction per Month using Custom Kernel')
%ylim([0 2])

%% Fit GPR with 10 years history and 3 years prediction by month
clc
N_weeks = floor(365/7);
Weeks = 1:N_weeks*10;
Weeks  = Weeks';

% Generate the monthly average data
for i = 1:13
    for j = 1:N_weeks
        Trs_week(j,i) = sum(Trs((j-1)*7+1:j*7,i))/7;
    end
end

T_series_week = [];
for i = 1:10
    T_series_week = [T_series_week;Trs_week(:,i)];
end

%Weeks_prd = 1:N_weeks*3;
%Weeks_prd = Weeks_prd';
% Days2 = 365+1:365*2;
% Days2 = Days2';

%TempMd = fitrgp(Trs(1:size(Tos),:),Tos);
% TempMd_week= fitrgp(Weeks,T_series_week,'KernelFunction','exponential');
% [ypred_week,~,yint_week] = predict(TempMd_week,Weeks_prd,'Alpha',0.1);
%ypred = forecast(TempMd,T_series,Days_prd);

%Trs_t = Trs((size(Tos)+1:365),:);
%[ypred1,~,yint1] = predict(TempMd,Trs_t);
D = size(Weeks,2);
%ypred = forecast(TempMd,T_series,Days_prd);
sigmaL10 = 0.1*ones(D(:,1),1);
sigmaL20 = 0.1;
sigmaF10 = 1;
sigmaF20 = 1;        
theta0   = [log(sigmaL10);log(sigmaL20);log(sigmaF10);log(sigmaF20)];

Weeks_prd = 10*N_weeks+1:13*N_weeks;
Weeks_prd = Weeks_prd';
hfcn = @(X)[X, sin(2*pi/N_weeks*X)];
beta0 = [1;1];
TempMd_week_custom= fitrgp(Weeks,T_series_week,'kernelfunction',@mykernal,'kernelparameters',theta0,'Basis',hfcn,'beta',beta0,'Sigma',2,'ComputationMethod','v');
[ypred_week_custom,~,yint_week_custom] = predict(TempMd_week_custom,Weeks_prd,'Alpha',0.1);


%New Plot for GPR by day
subplot(2,1,1)
T_new_week_custom = ypred_week_custom;
T_his_week_custom = [Trs_week(:,11);Trs_week(:,12);Trs_week(:,13)];
% X_week = 1:N_weeks*3;
% X_week = X_week';
X_week = datetime(2018,1,1) + calweeks(0:52*3-1);
X_week = X_week';
scatter(X_week,T_new_week_custom,'MarkerEdgeColor','#0072BD')
hold on
scatter(X_week,T_his_week_custom,'MarkerEdgeColor','#D95319')
patch([X_week;flipud(X_week)],[yint_week_custom(:,1);flipud(yint_week_custom(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
legend('Prediction','Real Data','90% Confidence Intervel','Location','southeast')
xlabel('Weeks')
ylabel('Temperature/K')
title('Prediction Result per Week using Custom Kernel')
%xlim([0 3*N_weeks])

subplot(2,1,2)
scatter(X_week,(T_new_week_custom-T_his_week_custom)./T_his_week_custom);
%xlim([0 3*N_weeks])
xlabel('Weeks')
ylabel('Error of Temperature/K')
title('Relative Error of Prediction per Week using Custom Kernel')


function KMN = mykernal(XM,XN,theta)
%UNTITLED2 Summary of this function goes here 
%   Detailed explanation goes here
D = size(XM,2);
params = exp(theta);
sigmaL1 = params(D+1,1);
sigmaF1 = params(D+2,1);
sigmaF2 = params(D+3,1);

% 3. Create the contribution due to squared exponential ARD.    
KMN = sin(pdist2(XM(:,1)*2*pi/12,XN(:,1)*2*pi/12)).^2;
for r = 2:D
  KMN = KMN + sin(pdist2(XM(:,r)*2*pi/12,XN(:,r)*2*pi/12)).^2;        
end
KMN = (sigmaF1^2)*exp(-2/sigmaL1*KMN);
% 4. Add the contribution due to squared exponential.
KMN = KMN + (sigmaF2^2)*exp(-2/sigmaL1*(sin(pdist2(XM*2*pi/12,XN*2*pi/12)).^2)); 
end