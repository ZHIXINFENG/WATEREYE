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

%% GPR Fit Each Year
Nyear = floor(size(T)/365);
for i = 1:Nyear
    Trs(:,i) = T(((i-1)*365+1):i*365);
end

Tos = T(i*365+1:size(T));

%% Define the basis function H


%% Fit GPR with 10 years history and 3 years prediction by day

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


Days_prd = 365*10+1:365*13;
Days_prd = Days_prd';

hfcn = @(X)[X,1.5*sin(2*pi/364*X)];
beta0 = [1;1];
TempMd = fitrgp(Days,T_series,'Basis',hfcn,'beta',beta0,'KernelFunction','rationalquadratic','Sigma',2,'ComputationMethod','v');
%TempMd = fitrgp(Days,T_series,'OptimizeHyperparameters','all')
[ypred,~,yint] = predict(TempMd,Days_prd,'Alpha',0.1);

%New Plot for GPR by day

subplot(2,1,1)
%T_new = ypred(365*10+1:365*13);
T_new = ypred;
T_his = [Trs(:,11);Trs(:,12);Trs(:,13)];  % History data
X = 1:365*3;
X = X';
scatter(X,T_new,'MarkerEdgeColor','#0072BD')
hold on
scatter(X,T_his,'MarkerEdgeColor','#D95319')
patch([X;flipud(X)],[yint(:,1);flipud(yint(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
%patch([X;flipud(X)],[yint(365*10+1:365*13,1);flipud(yint(365*10+1:365*13,2))],'k','FaceAlpha',0.1); % Prediction intervals
xlim([0 365*3])
legend('Prediction','Real Data','Intervel','Location','southeast')
xlabel('Days')
ylabel('Temperature/K')
title('Prediction Result per day')


subplot(2,1,2)
scatter(X,T_new-T_his);
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
D = size(T_series_month);
%TempMd = fitrgp(Trs(1:size(Tos),:),Tos);
%TempMd_months = fitrgp(Months,T_series_month,'KernelFunction','exponential');
%[ypred_month,~,yint_month] = predict(TempMd_months,Months_prd,'Alpha',0.1);
%[ypred_month,~,yint_month] = predict(TempMd_months,Months_prd,'Alpha',0.1);
%ypred = forecast(TempMd,T_series,Days_prd);
% H function x+sinx
Months_prd = 12*10+1:12*13;
Months_prd = Months_prd';
hfcn = @(X)[X, sin(2*pi/12*X)];
beta0 = [1;1];
TempMd_months = fitrgp(Months,T_series_month,'KernelFunction','rationalquadratic','Basis',hfcn,'beta',beta0,'Sigma',2,'ComputationMethod','v','Optimizer','fminsearch');
%TempMd_months = fitrgp(Months,T_series_month,'Basis',hfcn,'beta',beta0);
%TempMd_months = fitrgp(Months,T_series_month,'kernelfunction',@mykernel,'kernelparameters',theta0,'Basis',hfcn,'beta',beta0);
[ypred_month,~,yint_month] = predict(TempMd_months,Months_prd,'Alpha',0.1);

%Trs_t = Trs((size(Tos)+1:365),:);
%[ypred1,~,yint1] = predict(TempMd,Trs_t);

%New Plot for GPR by day
subplot(2,1,1)
T_new_months = ypred_month;
T_his_months = [Trs_month(:,11);Trs_month(:,12);Trs_month(:,13)];
% X = 1:12*3;
% X = X';
X_months = datetime(2018,1,1) + calmonths(0:12*3-1);
X_months = X_months';
scatter(X_months,T_new_months,'MarkerEdgeColor','#0072BD')
hold on
scatter(X_months,T_his_months,'MarkerEdgeColor','#D95319')
patch([X_months;flipud(X_months)],[yint_month(:,1);flipud(yint_month(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
legend('Prediction','Real Data','90% Confidence Intervel','Location','southeast')
xlabel('Months')
ylabel('Temperature/K')
title('Prediction Result per Month using Rationalquadratic Kernel ')
%xlim([0 36])

subplot(2,1,2)
X = X_months;
scatter(X_months,((T_new_months-T_his_months)./T_his_months));
%xlim([0 36])
xlabel('Months')
ylabel('Relative Error of Temperature')
title('Relative Error of Prediction Per Month using Rationalquadratic Kernel')

%% Fit GPR with 10 years history and 3 years prediction by weeks
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


Weeks_prd = 10*N_weeks+1:13*N_weeks;
Weeks_prd = Weeks_prd';
hfcn = @(X)[X, sin(2*pi/N_weeks*X)];
beta0 = [1;1];
TempMd_week= fitrgp(Weeks,T_series_week,'KernelFunction','rationalquadratic','Basis',hfcn,'beta',beta0,'Sigma',2,'ComputationMethod','v');
[ypred_week,~,yint_week] = predict(TempMd_week,Weeks_prd,'Alpha',0.1);


%New Plot for GPR by day
subplot(2,1,1)
T_new_week = ypred_week;
T_his_week = [Trs_week(:,11);Trs_week(:,12);Trs_week(:,13)];
% X_week = 1:N_weeks*3;
% X_week = X_week';
X_week = datetime(2018,1,1) + calweeks(0:52*3-1);
X_week = X_week';
scatter(X_week,T_new_week,'MarkerEdgeColor','#0072BD')
hold on
scatter(X_week,T_his_week,'MarkerEdgeColor','#D95319')
patch([X_week;flipud(X_week)],[yint_week(:,1);flipud(yint_week(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
legend('Prediction','Real Data','90% Confidence Intervel','Location','southeast')
xlabel('Weeks')
ylabel('Temperature/K')
title('Prediction Result per Week using Rationalquadratic Kernel')
%xlim([0 3*N_weeks])

subplot(2,1,2)
Tre_week = (T_new_week-T_his_week)./T_his_week;
scatter(X_week,Tre_week)
%xlim([0 3*N_weeks])
xlabel('Weeks')
ylabel('Relative Error of Temperature')
title('Relative Error of Prediction per Week using Rationalquadratic Kernel')

%% Plot monly results based on the average of the daily prediction

N_month = 36;

for i=1:N_month
    Tos_av(i) = sum(T_his((i-1)*30+1:i*30))/30;
end


for i = 1:N_month
    T_new_av(i) = sum(T_new((i-1)*30+1:i*30))/30;
end

% Interval average
for i = 1:N_month
    for j=1:2
        T_new_int_av(i,j) = sum(yint_new((i-1)*30+1:i*30,j))/30;
    end
end

subplot(2,1,1) 
X_av = 1:36;
X_av = X_av';
scatter(X_av,T_new_av,'MarkerEdgeColor','#0072BD')
hold on
scatter(X_av,Tos_av,'MarkerEdgeColor','#D95319')
patch([X_av;flipud(X_av)],[T_new_int_av(:,1);flipud(T_new_int_av(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
xlim([0 36])
xlabel('Months')
ylabel('Temperature/K')
title('Prediction Result per month')
legend('Prediction','Real Data','Intervel','Location','southeast')

subplot(2,1,2)
X2_av = 1:36;
scatter(X2_av,Tos_av(1:N_month)-T_new_av(1:N_month));
xlim([0 36])
xlabel('Months')
ylabel('Error of Temperature/K')
title('Error of prediction per month')


%% Plot weekly results based on the average of the daily prediction
N_week = floor(365*3/7);

for i=1:N_week
    Tos_av_w(i) = sum(T_his((i-1)*7+1:i*7))/7;
end

for i = 1:N_week
    T_new_av_w(i) = sum(T_new((i-1)*7+1:i*7))/7;
end

for i = 1:N_week
    for j=1:2
        T_new_int_av_w(i,j) = sum(yint_new((i-1)*7+1:i*7,j))/7;
    end
end

subplot(2,1,1)
X_av_w = 1:N_week;
X_av_w = X_av_w';
scatter(X_av_w,T_new_av_w,'MarkerEdgeColor','#0072BD')
hold on
scatter(X_av_w,Tos_av_w,'MarkerEdgeColor','#D95319')
patch([X_av_w;flipud(X_av_w)],[T_new_int_av_w(:,1);flipud(T_new_int_av_w(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
legend('Prediction','Real Data','Intervel','Location','southeast')
xlabel('Weeks')
ylabel('Temperature/K')
title('Prediction Result per week')
xlim([0 N_week])

subplot(2,1,2)
scatter(X_av_w,Tos_av_w(1:N_week)-T_new_av_w(1:N_week));
xlabel('Weeks')
ylabel('Error of Temperature/K')
title('Error of prediction per week')
xlim([0 N_week])


