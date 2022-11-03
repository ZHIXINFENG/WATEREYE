function uuout = EMPCController_Tower(currentx,currentV,currentPg,currentPw,t,currentA,currentB,predictedPw,currentTowerStatesFa1,currentTowerStatesFa2)
    global Parameters Solution
    persistent Controller

    % Hardcoded horizon length
    Np = 100;
    
    
    % Define terminal constraints
%         if currentV == 6
% %             add_constraint( x{Np+1} == Parameters.Je/2*59.217^2/Parameters.ScaleK );
%             TerminalC_K = Parameters.Je/2*59.217^2/Parameters.ScaleK;
%         end
%         if currentV == 7
% %             add_constraint( x{Np+1} == Parameters.Je/2*68.5966^2/Parameters.ScaleK );
%             TerminalC_K = Parameters.Je/2*68.5966^2/Parameters.ScaleK;
%         end
%         if currentV == 8
% %             add_constraint( x{Np+1} == Parameters.Je/2*78.1985^2/Parameters.ScaleK );
%             TerminalC_K = Parameters.Je/2*68.5966^2/Parameters.ScaleK;
%         end
%         if currentV == 9
% %             add_constraint( x{Np+1} == Parameters.Je/2*87.2139^2/Parameters.ScaleK );
%             TerminalC_K = Parameters.Je/2*87.2139^2/Parameters.ScaleK;
%         end
%         if currentV == 10
% %             add_constraint( x{Np+1} == Parameters.Je/2*95.7752^2/Parameters.ScaleK );
%             TerminalC_K = Parameters.Je/2*95.7752^2/Parameters.ScaleK;
%         end
%         if currentV == 11
% %             add_constraint( x{Np+1} == Parameters.Je/2*103.5882^2/Parameters.ScaleK );
%             TerminalC_K = Parameters.Je/2*103.5882^2/Parameters.ScaleK;
%         end
%         if currentV >11.4
% %             add_constraint( x{Np+1} == Parameters.Je/2*Parameters.Omegagrated^2/Parameters.ScaleK );
%               TerminalC_K = Parameters.Je/2*Parameters.Omegagrated^2/Parameters.ScaleK;
%         end

     predictedVs = currentV*ones(1,Np);
     predictedPw = predictedPw*ones(1,Np);
     predictedOmegag = sqrt(2/Parameters.Je*currentx)*ones(1,Np);
    %currentTowerStatesFa = [0.1 0.1 0.1]';

    if t == 0
        Solution.K = [];
        Solution.Pw = [];
        Solution.Pg = [];
        Solution.V  = [];
        Solution.Np = Np;
    end

    useIndividualForces = true;
    useTowerDynamicsAsConstraints = true;
    useIndAvailPower = true;

    % Drivetrain dynamics (in power and energy terms)
    Ad = 1; %K
    Bd = Parameters.Ts*[1 1 1 -1/Parameters.Effic]*Parameters.ScaleP/Parameters.ScaleK;%*Parameters.G^2; %[Pw1; Pw2; Pw3; Pg]
    Cd = [1];
    Dd = [0];

    reshapeDim = [1 Np];
%     predictedAzimuths = reshape(predictedAzimuths,reshapeDim);
    predictedVs = reshape(predictedVs,reshapeDim); 
    predictedPw = reshape(predictedPw,reshapeDim);
    predictedOmegag = reshape(predictedOmegag,reshapeDim);
    % be careful for the size of currentA and currentB
    pavCoefCol = 1;
    
%     Position1 = 1;
%     Position2 = 0.5;
%     [StaticPosition1] = PositionCompute(Parameters.ModeShape,Position1);
%     [StaticPosition2] = PositionCompute(Parameters.ModeShape,Position2);
    

    if t == 0
        tic
        currentTowerStatesFa1 = [0;0];
        currentTowerStatesFa2 = [0;0];
        currentx   = Parameters.Je/2*41.647046611088690^2;
%         currentx   = Parameters.Je/2*115^2;
%          currentx   = Parameters.J/2*32.6413^2;
        currentxx  = currentx/Parameters.ScaleK;
%         currentPgg = 2.4931e+05*Parameters.Effic/Parameters.ScaleP;
        currentPgg = 1e+05*Parameters.Effic/Parameters.ScaleP;
        currentPww = 2.4931e+05/Parameters.ScaleP; % entire rotor disk\
%         currentPgg = 3*10^5*Parameters.Effic/Parameters.ScaleP;
            
%         currentPww = 3*10^5/Parameters.ScaleP; % entire rotor disk\
%         currentPgg = 3.5*10^6*Parameters.Effic/Parameters.ScaleP;
%         currentPww = 3.5*10^6/Parameters.ScaleP; % entire rotor disk\
%         currentV   = 5.3 * (1+Parameters.R/Parameters.H*cos(predictedAzimuths(:,1))).^0.2;
%         currentA   = zeros(Parameters.Nconst,pavCoefCol);
%         currentB   = ones(Parameters.Nconst,pavCoefCol)*10;
        currentA = Parameters.Ainit;
        currentB = Parameters.Binit;
%         TerminalC_K = Parameters.Je/2*59.217^2/Parameters.ScaleK;
        
%         if useIndividualForces
%             predictedAzimuths(1,:) = Parameters.Ts * tril(ones(Np)) * Parameters.Omegagmin * sqrt(1.1)/Parameters.G*ones(Np,1);
%             predictedAzimuths(2,:) = predictedAzimuths(1,:) + ones(1,Np)*2*pi/3;
%             predictedAzimuths(3,:) = predictedAzimuths(2,:) + ones(1,Np)*2*pi/3;
%          currentV = 14;
            currentV = 6;      
           
        predictedVs = currentV*ones(1,Np);
%             predictedVs = currentV * (1+Parameters.R/Parameters.H*cos(predictedAzimuths)).^0.2;
%         [~,IndexV] = min((Parameters.BOUNDPW.V-currentV).^2);
%         predicted_theta = (predictedVs-Parameters.BOUNDPW.V(IndexV-1))./(Parameters.BOUNDPW.V(IndexV)-Parameters.BOUNDPW.V(IndexV-1));
%             
        predictedOmegag = 41.647046611088690*ones(1,Np);
%         predictedOmegag = 115*ones(1,Np);
%         predictedOmegag = 80*ones(1,Np);
        predictedPw    = currentPww*Parameters.ScaleP*ones(1,Np); % Aerodynamic power from the previous MPC solve
%             
        K_ = Parameters.Je/2 * predictedOmegag.^2;
        predictedPw=reshape(predictedPw,reshapeDim);
%         [betsa_ps,~,~] = calc_pitches(predictedOmegag, predictedVs, predictedPw, Np);
        [beta_ps,~,~] = calc_pitches(predictedOmegag, predictedVs, predictedPw);
        
        [Qo_,Ro_,So_] = linearize_force( beta_ps, K_, predictedVs, 'thrust' );
%         linear_out = linearize_force( beta_ps, K_, predictedVs, 'thrust' );
        
%         end
        
        % Avoid explosion of internally defined variables in YALMIP
        yalmip('clear')
        
        % Setup the optimization problem
        u       = sdpvar(repmat(2,1,Np),repmat(1,1,Np));
        x       = sdpvar(repmat(1,1,Np+1),repmat(1,1,Np+1));
        eps     = sdpvar(repmat(1,1,Np),repmat(1,1,Np));
        
        v       = sdpvar(pavCoefCol,1);
        Vp      = sdpvar(1,Np);
%         theta_Pav      = sdpvar(1,Np);
        AA      = sdpvar(Parameters.Nconst,pavCoefCol);
        BB      = sdpvar(Parameters.Nconst,pavCoefCol);
        PG      = sdpvar;
        PW      = sdpvar;
        PAV     = sdpvar(pavCoefCol,Np);
        
        Qo = sdpvar(repmat(1,1,Np),repmat(1,1,Np));
        Ro = sdpvar(repmat(1,1,Np),repmat(1,1,Np));
        So = sdpvar(repmat(1,1,Np),repmat(1,1,Np));
        Xtower_fa1 = sdpvar(repmat(2,1,Np+1),repmat(1,1,Np+1));
        Ytower_fa1 = sdpvar(repmat(3,1,Np),repmat(1,1,Np));
        Xtower_fa2 = sdpvar(repmat(2,1,Np+1),repmat(1,1,Np+1));
        Ytower_fa2 = sdpvar(repmat(3,1,Np),repmat(1,1,Np));
        Fo        = sdpvar(repmat(1,1,Np),repmat(1,1,Np));
        Ft        = sdpvar(repmat(1,1,Np),repmat(1,1,Np));
%         TC_K      = sdpvar;
        
        
        %% Setup Lp+QP
        constraints = [];
        objective = 0;
        
        for k = 1:Np
            % Maximize Pg
            add_objective( -1 * u{k}(2) ); % a1
            
%            add_objective( 10 * eps{k}^2 ); % 1e2 is OK  
            add_objective(  100 * eps{k}^2 ); % 1e2 is OK  %a5
            
            %add_objective( -10*PAV(1,k) );
            add_objective( -1*PAV(1,k) ); % a2

%             add_objective( 0.1 * Ytower_fa{k}(2).^2 );  
            
            if useIndividualForces
%                 add_objective( 0.1 * Ytower_fa1{k}(2).^2 );
%                 add_objective( 0.1 * Ytower_fa1{k}(2).^2 );
%                  add_objective( 0.1 * (Ytower_fa1{k}(2)+Ytower_fa2{k}(2)).^2 );
                  %add_objective( 1 * (Parameters.StaticPosition1(1)*Ytower_fa1{k}(2)).^2 );
                add_objective( 100 * (Parameters.StaticPosition1(1)*Ytower_fa1{k}(2)+Parameters.StaticPosition1(2)*Ytower_fa2{k}(2)).^2 );   %w1
                add_objective( 10 * (Parameters.StaticPosition2(1)*Ytower_fa1{k}(2)+Parameters.StaticPosition2(2)*Ytower_fa2{k}(2)).^2 );  %w2
%                 add_objective( 100 * (Parameters.StaticPosition1(1)*Ytower_fa1{k}(2)+Parameters.StaticPosition1(2)*Ytower_fa2{k}(2)).^2 );
%                 add_objective( 50 * (Parameters.StaticPosition2(1)*Ytower_fa1{k}(2)+Parameters.StaticPosition2(2)*Ytower_fa2{k}(2)).^2 );
            end
            
            
            
            % Euler
            % add_constraint( x{k+1} == Ad*x{k}+Bd*u{k} );
            
            % Tustin
            if k == 1
                add_constraint( x{k+1} == x{k} + ((u{k}(1)+PW(1)) - (u{k}(2)+PG)/Parameters.Effic)...
                    *Parameters.Ts/2*Parameters.ScaleP/Parameters.ScaleK );
            else
                add_constraint( x{k+1} == x{k} + ((u{k-1}(1)+u{k}(1))  - (u{k-1}(2)+u{k}(2))/Parameters.Effic)...
                    *Parameters.Ts/2*Parameters.ScaleP/Parameters.ScaleK );
            end
            
            add_constraint( 0 <= u{k}(2) <= Parameters.Effic*sqrt(2*x{k+1}*Parameters.ScaleK/Parameters.Je)*Parameters.Tmax/Parameters.ScaleP );
            add_constraint( 0 <= u{k}(2) <= 5e6/Parameters.ScaleP);
            
            add_constraint( Parameters.Je/2*Parameters.Omegagmin^2/Parameters.ScaleK <= (x{k+1}) <= Parameters.Je/2*Parameters.Omegagmax^2/Parameters.ScaleK );
            add_constraint( -Parameters.Je/2*Parameters.Omegagrated^2/Parameters.ScaleK <= x{k+1}-Parameters.Je/2*Parameters.Omegagrated^2/Parameters.ScaleK <= eps{k} );
            add_constraint( 0 <= eps{k} );
            
            if useIndAvailPower
                  add_constraint( 0<= (u{k}(1))<= PAV(1,k)*Vp(1,k)^3 );
                  add_constraint( PAV(1,k) <= min(AA(:)+BB(:)*x{k+1}) );   
% %                     add_constraint( 0<= (u{k}(1))<= (AA(:)+BB(:)*x{k+1})*v^3 ); 
% %                     PAV(k) = min(AA+BB*x{k+1})*v^3;
            else
                add_constraint( 0<= (u{k}(1))<= (AA(:)+BB(:)*x{k+1})*v^3 ); 
                PAV(k) = min(AA+BB*x{k+1})*v^3;
            end

%             theta_Pav(k) = (Vp(1,k)-Parameters.BOUNDPW.V(IndexV-1))/(Parameters.BOUNDPW.V(IndexV)-Parameters.BOUNDPW.V(IndexV-1));
%             add_constraint(0 <= u{k}(1) <= PAV(1,k)*Vp(1,k)^3);
%             add_constraint( PAV(1,k) <= min((1-theta_Pav(k))*(Parameters.BOUNDPW.Ac(:,IndexV-1)+Parameters.BOUNDPW.Bc(:,IndexV-1))*x{k+1}...
%                 +theta_Pav(k)*(Parameters.BOUNDPW.Ac(:,IndexV)+Parameters.BOUNDPW.Bc(:,IndexV)*x{k+1})));
            
            ratePenalty = 20;
            if k==1
                add_objective(ratePenalty*((u{k}(1)-PW)^2)/Parameters.Ts^2); %a4
                add_objective(ratePenalty*(u{k}(2)-PG)^2/Parameters.Ts^2);   %a3
            else
                add_objective(ratePenalty*((u{k}(1)-u{k-1}(1))^2)/Parameters.Ts^2);
                add_objective(ratePenalty*((u{k}(2)-u{k-1}(2))^2)/Parameters.Ts^2);
            end
            
            if useIndividualForces
                Fo{k}(1) = Qo{k}(1) * u{k}(1) * Parameters.ScaleP + Ro{k}(1) * x{k+1} * Parameters.ScaleK + So{k}(1) ;
                
                Ft{k}(1)   =  Fo{k}(1);
                
                if useTowerDynamicsAsConstraints
                    add_constraint( Xtower_fa1{k+1} == Parameters.sys_tower_fa_d.Am1*Xtower_fa1{k}+Parameters.sys_tower_fa_d.Bm1*Ft{k} );
                    add_constraint( Ytower_fa1{k} == Parameters.sys_tower_fa_d.Cm1*Xtower_fa1{k}+Parameters.sys_tower_fa_d.Dm1*Ft{k} );
                    add_constraint( Xtower_fa2{k+1} == Parameters.sys_tower_fa_d.Am2*Xtower_fa2{k}+Parameters.sys_tower_fa_d.Bm2*Ft{k} );
                    add_constraint( Ytower_fa2{k} == Parameters.sys_tower_fa_d.Cm2*Xtower_fa2{k}+Parameters.sys_tower_fa_d.Dm2*Ft{k} );
                else
                    Xtower_fa1{k+1} = Parameters.sys_tower_fa_d.Am1*Xtower_fa1{k}+Parameters.sys_tower_fa_d.Bm1*Ft{k};
                    Ytower_fa1{k} = Parameters.sys_tower_fa_d.Cm1*Xtower_fa1{k}+Parameters.sys_tower_fa_d.Dm1*Ft{k};
                    Xtower_fa2{k+1} = Parameters.sys_tower_fa_d.Am2*Xtower_fa2{k}+Parameters.sys_tower_fa_d.Bm2*Ft{k};
                    Ytower_fa2{k} = Parameters.sys_tower_fa_d.Cm2*Xtower_fa2{k}+Parameters.sys_tower_fa_d.Dm2*Ft{k};
                end
            end
        end
        add_objective( -x{Np+1} );
%         add_constraint( x{Np+1} == TC_K );

%         add_constraint( Xtower_fa1{Np+1} <= [0.0000001;0.00000001] );
%         add_constraint( Xtower_fa2{Np+1} <= [0.0000001;0.00000001] );
%          add_constraint( Xtower_fa1{Np+1} == [0;0] );
%          add_constraint( Xtower_fa2{Np+1} == [0;0] );
        % add_constraint( Xtower_fa2{Np} == [0;0;0] );
%         add_constraint( ((u{Np}(2)-u{Np-1}(2))^2)/Parameters.Ts^2 <= 1*10^(-6));
        
        
        ops = sdpsettings('solver','mosek','verbose',1,'debug',1);
        
%         parameters = {x{1},Vp,AA,BB,PG,PW,TC_K};
        parameters = {x{1},Vp,AA,BB,PG,PW};
        wantedVariables = {[u{:}],[x{:}],PAV,PG,PW};
        if useIndividualForces
            parameters = [parameters, {Xtower_fa1{1}},{Xtower_fa2{1}}];
%             parameters = [parameters, {Xtower_fa1{1}}];
            wantedVariables = [wantedVariables, {[Fo{:}],[Ft{:}],[Xtower_fa1{:}],[Ytower_fa1{:}],[Xtower_fa2{:}],[Ytower_fa2{:}]}];
%             wantedVariables = [wantedVariables, {[Fo{:}],[Ft{:}],[Xtower_fa1{:}],[Ytower_fa1{:}]}];
            parameters = [parameters, {[Qo{:}],[Ro{:}],[So{:}]}];
            wantedVariables = [wantedVariables, {[Qo{:}],[Ro{:}],[So{:}]}];
        end
        
        Controller = optimizer(constraints,objective,ops,parameters,wantedVariables);
        initElapsedTime = toc;
        fprintf('Initialization elapsed time = %d seconds.\n',initElapsedTime);
        tic
    else
%         [~,IndexV] = min((Parameters.BOUNDPW.V-currentV).^2);
%         predicted_theta = (predictedVs-Parameters.BOUNDPW.V(IndexV-1))./(Parameters.BOUNDPW.V(IndexV)-Parameters.BOUNDPW.V(IndexV-1));
        currentxx  = currentx/Parameters.ScaleK;
        currentPgg = currentPg/Parameters.ScaleP;
        currentPww = currentPw/Parameters.ScaleP; % TODO Pw should be Np by 3
        
        predictedPw=reshape(predictedPw,reshapeDim);
        [beta_ps,~,~] = calc_pitches(predictedOmegag, predictedVs, predictedPw);
   
        K_ = Parameters.Je/2 * predictedOmegag.^2;
        [Qo_,Ro_,So_] = linearize_force( beta_ps, K_, predictedVs, 'thrust' );
    end
    
%     if useIndAvailPower && (length(currentV) == 1)
%         currentV = currentV * ones(3,1);
%     end
    
%     inputParameters = {currentxx,predictedVs,currentA,currentB,currentPgg,currentPww,TerminalC_K};
        inputParameters = {currentxx,predictedVs,currentA,currentB,currentPgg,currentPww};
%       inputParameters = {currentxx,predictedVs,currentPgg,currentPww};
    if useIndividualForces
        inputParameters = [inputParameters, {currentTowerStatesFa1},{currentTowerStatesFa2}];
%         inputParameters = [inputParameters, {currentTowerStatesFa1}];
        inputParameters = [inputParameters, {Qo_,Ro_,So_}];
        
        if any(isnan(currentxx)) || any(isnan(predictedVs),'all') || any(isnan(currentA),'all') ...
            || any(isnan(currentB),'all') || any(isnan(currentPgg)) || any(isnan(currentPww),'all')...
            || any(isnan(predictedOmegag),'all')
        warning('There''s NAN!!!');
        end
    end
    
%     tic
    % And use it here too
    [ uout_complete] = Controller{inputParameters};
    solveElapsedTime = toc;
    fprintf('Solve elapsed time = %d seconds.\n',solveElapsedTime)
    
    PwComplete    = uout_complete{1}(1,:) * Parameters.ScaleP;
    PgComplete    = uout_complete{1}(2,:)   * Parameters.ScaleP;
    KComplete     = uout_complete{2}        * Parameters.ScaleK;
    
    Pw     = PwComplete(1);
    Pg     = PgComplete(1);
    

    Pavail = uout_complete{3}(1)*Parameters.ScaleP;
    
    omegaGComplete = sqrt(KComplete*2/Parameters.Je);
    omegaG         = omegaGComplete(2); % Should be the 2nd element, since the 1st one is known before the solve
    
    % Anticipate NaNs
    omegaG = bypassNaN(omegaG,sqrt(currentx*Parameters.ScaleK*2/Parameters.Je));
    Pw     = bypassNaN(Pw,currentPww*Parameters.ScaleP);
    Pg     = bypassNaN(Pg,currentPgg*Parameters.ScaleP);
    
    Tg     = Pg/omegaG;
    if useIndividualForces
        FoOut  = uout_complete{6}(1);% * Parameters.ScaleP/sqrt(Parameters.ScaleK);
        FtOut = uout_complete{7}(1);% * Parameters.ScaleP/sqrt(Parameters.ScaleK);
        XfaOut1 = uout_complete{9}(1);
        VfaOut1 = uout_complete{9}(2);
        AfaOut1 = uout_complete{9}(3);
        XfaOut2 = uout_complete{11}(1);
        VfaOut2 = uout_complete{11}(2);
        AfaOut2 = uout_complete{11}(3);
        QiOut = uout_complete{12}(1);
        RiOut = uout_complete{13}(1);
        SiOut = uout_complete{14}(1);
            VfaOut1Complete = uout_complete{9}(2,:);
    VfaOut2Complete = uout_complete{11}(2,:);
%         QiOut = uout_complete{10}(1);
%         RiOut = uout_complete{11}(1);
%         SiOut = uout_complete{12}(1);
    end

    
    
    % compute pitch
    [Pitch,TSR,Cpp] = calc_pitch(omegaG, currentV, Pw);
    
    % Dimension: [1 1 1 1 1 1 1 100 100 1 1 1 1 1 1 3 1 1 1 1 1 1 1]
%     uuout=[Pg; Tg; Pw(:); Pitch(:); TSR(:); Cpp(:); omegaG; omegaGComplete(2:end)'; PwComplete(:); Pavail; 0;0;0;0;0;0;0;0;solveElapsedTime];
    uuout=[Pg; Tg; Pw(:); Pitch(:); TSR(:); Cpp(:); omegaG; omegaGComplete(2:end)';PwComplete(:);PgComplete(:);...
           VfaOut1Complete(:); VfaOut2Complete(:); Pavail; 0;0;0;0;0;0;0;0;0;0;0;solveElapsedTime];
    if useIndividualForces
        uuout(end-11:end-4) = [FoOut;FtOut;XfaOut1;VfaOut1;AfaOut1;XfaOut2;VfaOut2;AfaOut2];
%         uuout(end-11:end-7) = [FoOut;FtOut;XfaOut1;VfaOut1;AfaOut1];
        uuout(end-3:end-1) = [QiOut; RiOut; SiOut];
    end
    uuout(isnan(uuout)) = 0;
    
    Solution.Pw =  [Solution.Pw PwComplete']; % [Np x 3, Np x 3, ... Np x 3]
    Solution.Pg  = [Solution.Pg PgComplete'];
    Solution.K   = [Solution.K  KComplete'];
    Solution.V = [Solution.V predictedVs'];

    function add_constraint( newConstraint )
        if( ~exist('constraints','var') ) 
            constraints = newConstraint;
        else
            constraints = [ constraints, newConstraint ];
        end        
    end

    function add_objective( newObjective )
        if( ~exist('objective','var') )                     
            objective = newObjective;
        else
            objective = objective + newObjective;
        end        
    end
end

%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%
function newValue = bypassNaN(newValue,oldValue)
    if isnan(newValue)
        newValue = oldValue;
    end
end

function [Pitch,TSR,Cpp] = calc_pitches(omega_g, currentV, Pw)
global Parameters
pitch = 0:0.005:25;
TSR=omega_g/Parameters.G*Parameters.R./currentV;
Cpp=Pw./(1/2*Parameters.rho*pi*Parameters.R^2*currentV.^3);
Cp = max(interp2(Parameters.Rotor_Lamda,Parameters.Rotor_Pitch,Parameters.Rotor_cP,TSR(:),pitch,'linear'),0);
[MinObje,IndexPitch]=min((Cp-Cpp(:)').^2);
Pitch=pitch(IndexPitch);
end

function [Pitch,TSR,Cpp] = calc_pitch(omega_g, currentV, Pw)
global Parameters
pitch=0:0.005:20;
TSR=omega_g/Parameters.G*Parameters.R/currentV;
Cpp=Pw/(1/2*Parameters.rho*pi*Parameters.R^2*currentV^3);
Cp = max(interp2(Parameters.Rotor_Lamda,Parameters.Rotor_Pitch,Parameters.Rotor_cP,TSR,pitch,'cubic'),0);
[~,idxPitch]=min((Cp-Cpp).^2);
Pitch=pitch(idxPitch);
end