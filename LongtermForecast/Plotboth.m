%% Monthly Prediction using custom kernel
Months_30years = 12*10+1:12*30;
Months_30years = Months_30years';
[ypred_30years,~,yint_30years] = predict(TempMd_months,Months_30years,'Alpha',0.1);
[ypred_train_month,~,yint_train_month] = predict(TempMd_months,Months,'Alpha',0.1);
[ypred_30years_custom,~,yint_30years_custom] = predict(TempMd_months_custom,Months_30years,'Alpha',0.1);
[ypred_train_month_custom,~,yint_train_month_custom] = predict(TempMd_months_custom,Months,'Alpha',0.1);

%% Plot for month custom kernel

subplot(2,1,1)
X_months = datetime(2008,1,1) + calmonths(0:12*10-1);
X_months_30years =  datetime(2008,1,1) + calmonths(0:12*30-1);
X_months_prd = datetime(2018,1,1) + calmonths(0:12*3-1);
%yint_month_total(:,1) = [yint_train_month(:,1);yint_30years_custom(:,1)];
%yint_month_total(:,2) = [yint_train_month(:,2);yint_30years_custom(:,2)];
yint_month_total(:,1) = [yint_train_month(:,1);yint_month(:,1)];
yint_month_total(:,2) = [yint_train_month(:,2);yint_month(:,2)];
plot([X_months';X_months_prd'],[ypred_train_month;ypred_month],'MarkerEdgeColor','#7E2F8E')
hold on
plot(X_months,T_series_month,'o','MarkerSize',12,'MarkerEdgeColor','#EDB120')
hold on
plot(X_months_prd,T_his_months,'o','MarkerSize',12,'MarkerEdgeColor','#A2142F')
hold on
patch([[X_months';X_months_prd'];flipud([X_months';X_months_prd'])],[yint_month_total(:,1);flipud(yint_month_total(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
legend('Prediction','Train set','Test Set','Confidence Intervel','Location','southeast')
xlabel('Months')
ylabel('Temperature/K')
title('Prediction Result per Month using Rationalquadratic Kernel ')

subplot(2,1,2)
plot(X_months',ypred_train_month-T_series_months,'Color','#0072BD')
hold on 
plot(X_months_prd',T_new_month-T_his_month,'Color','#0072BD')


% X = [Months;Months_30years];
% 
% patch([X;flipud(X)],[yint(:,1);flipud(yint(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
% xlim([0 12*30])

%% Days plot
% Derive the train set data
Days_30years = 365*10+1:365*30;
Days_30years = Days_30years';
[ypred_Days_30years,~,yint__Days_30years] = predict(TempMd,Days_30years,'Alpha',0.1);
[ypred_train_Days,~,yint_train_Days] = predict(TempMd,Days,'Alpha',0.1);
%% Custom Prediction
[ypred_Days_30years_custom,~,yint__Days_30years_custom] = predict(TempMd_custom,Days_30years,'Alpha',0.1);
[ypred_train_Days_custom,~,yint_train_Days_custom] = predict(TempMd_custom,Days,'Alpha',0.1);

%% Plot
X_Days = datetime(2008,1,1) + caldays(1:365*10);
X_Days_prd = datetime(2018,1,1) + caldays(1:365*3);
X_Days_30years = datetime(2008,1,1) + caldays(1:365*30);
plot(X_Days,T_series,'.','MarkerSize',12,'MarkerEdgeColor','#EDB120')
hold on
plot(X_Days_prd,T_his,'.','MarkerSize',12,'MarkerEdgeColor','#A2142F')
hold on
plot(X_Days_30years,[ypred_train_Days;ypred_Days_30years],'Color','#0072BD')
hold on
plot(X_Days_30years,[ypred_train_Days_custom;ypred_Days_30years_custom],'Color','#7E2F8E')
hold on
plot(X_Days_30years,ANN(1:10950)-273.15,'Color','#77AC30')
legend('Train Set','Test Set','Rationalquadratic Kernel', 'Custom Kernel','PINN Model','Location','southeastoutside')
xlabel('Time /Year')
ylabel('Temperature/Degree')
title('Comparison of Prediction Result per day')

%% Error
plot([X_Days';X_Days_prd'],[ypred_train_Days-T_series;T_new-T_his],'Color','#0072BD')
hold on 
hold on
plot([X_Days';X_Days_prd'],[ypred_train_Days_custom-T_series;T_new_custom-T_his],'Color','#7E2F8E')
hold on 
plot([X_Days';X_Days_prd'],[ANN(1:365*10)-273.15-T_series;ANN(365*10+1:365*13)-273.15-T_his],'Color','#77AC30')
legend('Rationalquadratic Kernel', 'Custom Kernel','PINN Model','Location','northeast')
xlabel('Time /Year')
ylabel('Error/Degree')
title('Comparison of Error of Daily Prediction')
