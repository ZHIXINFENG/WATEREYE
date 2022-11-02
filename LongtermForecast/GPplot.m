%% Preprocessing
T_1 = Temp(1,:);
T_1 = T_1';
Time = 1:4890;
Time = Time';
Time_h = 1:4500;
Time_t = 4501:4890;
Time_h = Time_h';
Time_t = Time_t';
T_h = Temp(:,1:4500);
T_t = Temp(:,4501:4890);
T_h1 = T_h(1,:);
T_h1 = T_h1';
T_t1 = T_t(1,:);
T_t1 = T_t1';

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

Tos = T(i*365+1:size(T));

%% Fit GPR with 10 years history and 3 years prediction

Days = 1:365;
Days = Days';

% Days2 = 365+1:365*2;
% Days2 = Days2';

%TempMd = fitrgp(Trs(1:size(Tos),:),Tos);
TempMd = fitrgp([Trs(:,1:9) Days],Trs(:,10),'KernelFunction','squaredexponential');
[ypred1,~,yint1] = predict(TempMd,[Trs(:,2:10) Days]);
[ypred2,~,yint2] = predict(TempMd,[Trs(:,2:10) Days]);
[ypred3,~,yint3] = predict(TempMd,[Trs(:,2:10) Days]);
%Trs_t = Trs((size(Tos)+1:365),:);
%[ypred1,~,yint1] = predict(TempMd,Trs_t);

%New Plot for GPR by day

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

%% Fit GPR with 10 years history and 3 years prediction

Days = 1:365*10;
Days = Days';

T_series = [];
for i = 1:10
    T_series = [T_series;Trs(:,i)];
end

Days_prd = 1:365*3;
Days_prd = Days_prd'+0.1;
% Days2 = 365+1:365*2;
% Days2 = Days2';

%TempMd = fitrgp(Trs(1:size(Tos),:),Tos);
TempMd = fitrgp(Days,T_series,'KernelFunction','squaredexponential');
[ypred,~,yint] = predict(TempMd,Days_prd,'Alpha',0.1);
%ypred = forecast(TempMd,T_series,Days_prd);

%Trs_t = Trs((size(Tos)+1:365),:);
%[ypred1,~,yint1] = predict(TempMd,Trs_t);

%New Plot for GPR by day

T_new = ypred;
T_his = [Trs(:,11);Trs(:,12);Trs(:,13)];
X = 1:365*3;
X = X';
scatter(X,T_new,'MarkerEdgeColor','#0072BD')
hold on
X2 = 1:145
scatter(X,T_his,'MarkerEdgeColor','#D95319')
patch([X;flipud(X)],[yint(:,1);flipud(yint(:,2))],'k','FaceAlpha',0.1); % Prediction intervals

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