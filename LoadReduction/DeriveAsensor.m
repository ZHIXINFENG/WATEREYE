TowerDynamics.Asensor1.Time = TowerDynamics.Amodal1.Time;
TowerDynamics.Asensor2.Time = TowerDynamics.Amodal1.Time;
TowerDynamics.Asensor1.Data = TowerDynamics.Amodal1.Data*Parameters.StaticPosition1(1)+ TowerDynamics.Amodal2.Data*Parameters.StaticPosition1(2);
TowerDynamics.Asensor2.Data = TowerDynamics.Amodal1.Data*Parameters.StaticPosition2(1)+ TowerDynamics.Amodal2.Data*Parameters.StaticPosition2(2);