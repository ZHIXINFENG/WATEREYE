%% Np Paramatric Analysis
load('Np50.mat','TowerDynamics');
TowerDynamics_50 = TowerDynamics;
load('Np100.mat','TowerDynamics');
TowerDynamics_100 = TowerDynamics;
load('Np200.mat','TowerDynamics');
TowerDynamics_200 = TowerDynamics;

Tcompute_50 = TowerDynamics_50.EpliseTime.Data(2:end)-TowerDynamics_50.EpliseTime.Data(1:end-1);
Tcompute_50(2:length(Tcompute_50)+1) = Tcompute_50;
Tcompute_50(1) = TowerDynamics_50.EpliseTime.Data(1);

Tcompute_100 = TowerDynamics_100.EpliseTime.Data(2:end)-TowerDynamics_100.EpliseTime.Data(1:end-1);
Tcompute_100(2:length(Tcompute_100)+1) = Tcompute_100;
Tcompute_100(1) = TowerDynamics_100.EpliseTime.Data(1);

Tcompute_200 = TowerDynamics_200.EpliseTime.Data(2:end)-TowerDynamics_200.EpliseTime.Data(1:end-1);
Tcompute_200(2:length(Tcompute_200)+1) = Tcompute_200;
Tcompute_200(1) = TowerDynamics_200.EpliseTime.Data(1);

%% Plot
plot(Tcompute_200)
hold on
plot(Tcompute_100)
hold on
plot(Tcompute_50)
xlim([0 6000])

%% Plot Average

v=6:1:17;

for i=1:floor(length(Tcompute_50)/(100/0.2))
    T_mean_50(i) = mean(Tcompute_50((i-1)*500+1:i*500));
    T_mean_100(i) = mean(Tcompute_100((i-1)*500+1:i*500));
    T_mean_200(i) = mean(Tcompute_200((i-1)*500+1:i*500));
end

f = figure;
f.Position(3:4) = [210 200];
% subplot(1,2,1)
stairs(v,T_mean_50,'Color','black','LineStyle','-','LineWidth',1)
hold on
stairs(v,T_mean_100,'Color','black','LineStyle','--','LineWidth',1)
hold on
stairs(v,T_mean_200,'Color','black','LineStyle',':','LineWidth',1)
grid on
xlim([6 17])
xlabel('$v_\mathrm{w}$ (m/s)','Interpreter','latex','FontSize',8)
ylabel('$T_{\mathrm{compute}}$ (s)','Interpreter','latex','FontSize',8)
%legend('$N_p$=10s','$N_p$=20s','$N_p$=40s','Interpreter','latex','FontSize',8)
set(gcf, 'Color', 'w');


%% 
%time_50 = 1:50;
%time_100 =1:100;
%time_200 =1:200;
%subplot(3,2,1)
%plot(time_50,TowerDynamics_50.PgComplete.Data(20,:)/Parameters.ScaleP)
%hold on
%plot(time_100,TowerDynamics_100.PgComplete.Data(20,:)/Parameters.ScaleP)
%hold on
%plot(time_200,TowerDynamics_200.PgComplete.Data(20,:)/Parameters.ScaleP)
% f = figure;
% f.Position(3:4) = [1000 250];
f = figure;
f.Position(3:4) = [210 200];
time = 0:0.2:1200;
% subplot(1,2,2)
plot(time,TowerDynamics_50.Pg.Data,'Color','black','LineStyle','-','LineWidth',1)
hold on
plot(time,TowerDynamics_100.Pg.Data,'Color','black','LineStyle','--','LineWidth',1)
hold on
grid on
plot(time,TowerDynamics_200.Pg.Data,'Color','black','LineStyle',':','LineWidth',1)
xlabel('Time (s)','Interpreter','latex','FontSize',8)
ylabel('$P_\mathrm{g}$ (W)','Interpreter','latex','FontSize',8)
legend('$N_p$=50','$N_p$=100','$N_p$=200','Interpreter','latex','FontSize',8)
xlim([0 1200])
%ylim([0 7.5*10^6])

set(gcf, 'Color', 'w');

%plot(time_50,(Parameters.Je/2*TowerDynamics_50.signal8.Data(20,:)^2-Parameters.Je/2*Parameters.Omegagrated^2)/Parameters.ScaleK)
