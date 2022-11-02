%% This code is based on the prediction for each year, without time sequence in year, which is the version I explain this afternoon

%% Preprocessing
clear
clc
load('Temperature.mat');
T_1 = Temp(1,:); % Becasue we have the data at 0, 6, 12, 18, here we use the data at 0 hour as everyday's measurement
T_1 = T_1';

Days = 1:365;
Days = Days';
%% Preprocessing 2
T = Temp(1,:);
T = T';

%% GPR Fit Each Year
Nyear = floor(size(T)/365);
for i = 1:Nyear
    Trs(:,i) = T(((i-1)*365+1):i*365);
end

%Tos = T(i*365+1:size(T));

%% Fit GPR with 10 years history and 3 years prediction

Days = 1:365;
Days = Days';


TempMd = fitrgp([Trs(:,1:9) Days],Trs(:,10),'KernelFunction','squaredexponential');
[ypred1,~,yint1] = predict(TempMd,[Trs(:,2:10) Days],'Alpha',0.1);
[ypred2,~,yint2] = predict(TempMd,[Trs(:,2:10) Days],'Alpha',0.12); % Assuming the significance increase by year
[ypred3,~,yint3] = predict(TempMd,[Trs(:,2:10) Days],'Alpha',0.15);


T_new = [ypred1;ypred2;ypred3];
yint_new = [yint1;yint2;yint3];
T_his = [Trs(:,11);Trs(:,12);Trs(:,13)];
X = 1:365*3;
X = X';
scatter(X,T_new,'MarkerEdgeColor','#0072BD')
hold on
%X2 = 1:145
scatter(X,T_his,'MarkerEdgeColor','#D95319')
patch([X;flipud(X)],[yint_new(:,1);flipud(yint_new(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
hold off

%% Fit GPR with 10 years history and 3 years prediction by month

Months = 1:12;
Months = Months';
N_month = 12;
% Generate the monthly average data
for i = 1:13
    for j = 1:N_month
        Trs_month(j,i) = sum(Trs((j-1)*30+1:j*30,i))/30;
    end
end

Months_prd = 1:12;
Months_prd = Months_prd';
% Days2 = 365+1:365*2;
% Days2 = Days2';

%TempMd = fitrgp(Trs(1:size(Tos),:),Tos);
TempMd_months = fitrgp([Trs_month(:,1:9),Months],Trs_month(:,10),'KernelFunction','exponential');
[ypred_month1,~,yint_month1] = predict(TempMd_months,[Trs_month(:,2:10), Months_prd],'Alpha',0.1);
[ypred_month2,~,yint_month2] = predict(TempMd_months,[Trs_month(:,2:10), Months_prd],'Alpha',0.12);
[ypred_month3,~,yint_month3] = predict(TempMd_months,[Trs_month(:,2:10), Months_prd],'Alpha',0.15);
%[ypred_month,~,yint_month] = predict(TempMd_months,Months_prd,'Alpha',0.1);
%ypred = forecast(TempMd,T_series,Days_prd);

%Trs_t = Trs((size(Tos)+1:365),:);
%[ypred1,~,yint1] = predict(TempMd,Trs_t);

%New Plot for GPR by day
subplot(2,1,1)
T_new_months = [ypred_month1;ypred_month2;ypred_month3];
T_his_months = [Trs_month(:,11);Trs_month(:,12);Trs_month(:,13)];
yint_new_months = [yint_month1;yint_month2;yint_month3];
X = 1:12*3;
X = X';
scatter(X,T_new_months-273.15,'MarkerEdgeColor','#0072BD')
hold on
scatter(X,T_his_months-273.15,'MarkerEdgeColor','#D95319')
patch([X;flipud(X)],[yint_new_months(:,1)-273.15;flipud(yint_new_months(:,2)-273.15)],'k','FaceAlpha',0.1); % Prediction intervals
legend('Prediction','Real Data','Intervel','Location','southeast')
xlabel('Months')
ylabel('Temperature/Degree')
title('Prediction Result per week')
xlim([0 36])

subplot(2,1,2)
scatter(X,T_new_months-T_his_months);
xlim([0 36])
xlabel('Months')
ylabel('Error of Temperature/K')
title('Error of prediction per month')

%% Fit GPR with 10 years history and 3 years prediction by weeks
N_weeks = floor(365/7);
Weeks = 1:N_weeks;
Weeks  = Weeks';

% Generate the monthly average data
for i = 1:13
    for j = 1:N_weeks
        Trs_week(j,i) = sum(Trs((j-1)*7+1:j*7,i))/7;
    end
end

Weeks_prd = 1:N_weeks;
Weeks_prd = Weeks_prd';
% Days2 = 365+1:365*2;
% Days2 = Days2';

TempMd_weeks = fitrgp([Trs_week(:,1:9),Weeks],Trs_week(:,10),'KernelFunction','exponential');
[ypred_week1,~,yint_week1] = predict(TempMd_weeks,[Trs_week(:,2:10), Weeks_prd],'Alpha',0.1);
[ypred_week2,~,yint_week2] = predict(TempMd_weeks,[Trs_week(:,2:10), Weeks_prd],'Alpha',0.12);
[ypred_week3,~,yint_week3] = predict(TempMd_weeks,[Trs_week(:,2:10), Weeks_prd],'Alpha',0.15);



%New Plot for GPR by day
subplot(2,1,1)
T_new_week = [ypred_week1;ypred_week2;ypred_week3];
yint_week = [yint_week1;yint_week2;yint_week3];
T_his_week = [Trs_week(:,11);Trs_week(:,12);Trs_week(:,13)];
X_week = 1:N_weeks*3;
X_week = X_week';
scatter(X_week,T_new_week,'MarkerEdgeColor','#0072BD')
hold on
scatter(X_week,T_his_week,'MarkerEdgeColor','#D95319')
patch([X_week;flipud(X_week)],[yint_week(:,1);flipud(yint_week(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
legend('Prediction','Real Data','Intervel','Location','southeast')
xlabel('Weeks')
ylabel('Temperature/K')
title('Prediction Result per week')
xlim([0 N_weeks])

subplot(2,1,2)
scatter(X_week,T_new_week-T_his_week);
xlim([0 N_weeks])
xlabel('Months')
ylabel('Error of Temperature/K')
title('Error of prediction per month')



%% Plot monly average value

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


%% Plot weekly everage value
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


%% Fit GPR by day
Days = 1:365;
Days = Days';

%TempMd = fitrgp(Trs(1:size(Tos),:),Tos);
TempMd = fitrgp([Trs(:,1:12) Days],Trs(:,13),'KernelFunction','squaredexponential');
[ypred1,~,yint1] = predict(TempMd,[Trs(:,2:13) Days]);
%Trs_t = Trs((size(Tos)+1:365),:);
%[ypred1,~,yint1] = predict(TempMd,Trs_t);

%New Plot for GPR by day

T_new = ypred1;
X = 1:365;
scatter(X,T_new)
hold on
X2 = 1:145
scatter(X2,Tos)

%% Plot monly average value

N_month = floor(size(Tos)/30);

for i=1:N_month
    Tos_av(i) = sum(Tos((i-1)*30+1:i*30))/30;
end


for i = 1:12
    T_new_av(i) = sum(Trs((i-1)*30+1:i*30))/30;
end

subplot(1,2,1)
X_av = 1:12;
scatter(X_av,T_new_av)
hold on
X2_av = 1:N_month;
scatter(X2_av,Tos_av)
xlabel('Months')
ylabel('Temperature/K')
title('Prediction Result per month')

subplot(1,2,2)
X2_av = 1:N_month;
scatter(X2_av,Tos_av(1:N_month)-T_new_av(1:N_month));
xlabel('Months')
ylabel('Error of Temperature/K')
title('Error of prediction per month')


%% Plot weekly everage value
N_week = floor(size(Tos)/7);

for i=1:N_week
    Tos_av_w(i) = sum(Tos((i-1)*7+1:i*7))/7;
end

N_week_total = floor(365/7);

for i = 1:N_week_total
    T_new_av_w(i) = sum(Trs((i-1)*7+1:i*7))/7;
end

subplot(1,2,1)
X_av_w = 1:N_week_total;
scatter(X_av_w,T_new_av_w)
hold on
X2_av_w = 1:N_week;
scatter(X2_av_w,Tos_av_w)
xlabel('Weeks')
ylabel('Temperature/K')
title('Prediction Result per week')

subplot(1,2,2)
scatter(X2_av_w,Tos_av_w(1:N_week)-T_new_av_w(1:N_week));
xlabel('Weeks')
ylabel('Error of Temperature/K')
title('Error of prediction per week')

%% GPR Fit
%TempMd = fitrgp(Time,T_1);
%[ypred1,~,yint1] = predict(TempMd,Time);
%TempMd = fitrgp(Time_h,T_h1,'CrossVal','on');
TempMd = fitrgp(Time_h,T_h1,'KernelFunction','ardsquaredexponential',...
      'FitMethod','fic','PredictMethod','fic','Standardize',1);
%TempMd_cv = crossval(TempMd);
[ypred1,~,yint1] = predict(TempMd,Time);
%ypred1 = forecast(TempMd,T_h1,Time_t);
%ypred1 = predict(TempMd,[Time_h;zeros(390,1)],390);


%% Plot
nexttile
hold on
%scatter(Time_h,T_h1,'r') % Observed data points
%hold on
scatter(Time_t,T_t1,'r')
scatter(Time_t,ypred1(4501:4890,1),'g')   % GPR predictions
%patch([x;flipud(x)],[yint1(:,1);flipud(yint1(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
hold off
title('GPR Fit of Temperature')
legend({'GPR predictions','95% prediction intervals'},'Location','best')