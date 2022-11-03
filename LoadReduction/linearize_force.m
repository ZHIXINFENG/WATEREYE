function [Q,R,S] = linearize_force(pitch_bar,K_bar,v,force_type,varargin)
global Parameters
length(varargin)
switch length(varargin)
    case 2
        interpMethod = varargin{1};
        isBladeEffective = varargin{2};
    case 1
        interpMethod = varargin{1};
        isBladeEffective = true;
    otherwise
        interpMethod = 'spline';
        isBladeEffective = true;
end

if v == 0
    Q = 0;
    R = 0;
    S = 0;
else
    omega_r = sqrt(2*K_bar/Parameters.Je) / Parameters.G;
    lambda_bar = omega_r * Parameters.R ./ v;
    
    for i=1:size(lambda_bar)
        if lambda_bar(i) > 10
            lambda_bar(i) = 10;
        end
    end
    
    dlambda_domegar = Parameters.R./v;
    domegar_dK = 1/2 * sqrt(2./(Parameters.Je*K_bar))/Parameters.G;

    % dCp/dK = dCp/dlambda dlambda/dwr * dwr/dK

    % Power coefficient
    dCp_dlambda = interp2(Parameters.Rotor_Lamda, Parameters.Rotor_Pitch, Parameters.dCp_dlambda, lambda_bar, pitch_bar, interpMethod);
    qp          = interp2(Parameters.Rotor_Lamda, Parameters.Rotor_Pitch, Parameters.dCp_dbeta, lambda_bar, pitch_bar, interpMethod); % basically dCp_dbeta
    rp          = dCp_dlambda .* dlambda_domegar .* domegar_dK; % basically dCp_dK
    Cp          = interp2(Parameters.Rotor_Lamda, Parameters.Rotor_Pitch, Parameters.Rotor_cP, lambda_bar, pitch_bar, interpMethod);
    sp          = Cp - qp .* pitch_bar - rp .* K_bar;

    % Force
    switch force_type
        case {'thrust', 'outofplane'}
            dCf_dlambda = interp2(Parameters.Rotor_Lamda, Parameters.Rotor_Pitch, Parameters.dCt_dlambda, lambda_bar, pitch_bar, interpMethod);  
            qf          = interp2(Parameters.Rotor_Lamda, Parameters.Rotor_Pitch, Parameters.dCt_dbeta, lambda_bar, pitch_bar, interpMethod); % basically dCf_dbeta
            rf          = dCf_dlambda .* dlambda_domegar .* domegar_dK; % basically dCf_dK
            Cf          = interp2(Parameters.Rotor_Lamda, Parameters.Rotor_Pitch, Parameters.Rotor_cT, lambda_bar, pitch_bar, interpMethod);
            sf          = Cf - qf .* pitch_bar - rf .* K_bar;
        case 'inplane'
            dCf_dlambda = interp2(Parameters.Rotor_Lamda, Parameters.Rotor_Pitch, Parameters.dCq_dlambda, lambda_bar, pitch_bar, interpMethod);  
            qf          = interp2(Parameters.Rotor_Lamda, Parameters.Rotor_Pitch, Parameters.dCq_dbeta, lambda_bar, pitch_bar, interpMethod);  
            rf          = dCf_dlambda .* dlambda_domegar .* domegar_dK; % basically dCf_dK
            Cf          = interp2(Parameters.Rotor_Lamda, Parameters.Rotor_Pitch, Parameters.Rotor_cQ, lambda_bar, pitch_bar, interpMethod);
            sf          = Cf - qf .* pitch_bar - rf .* K_bar;
        otherwise
            error('Your ''force_type'' is incorrect! Please choose ''thrust'', ''inplane'', or ''outofplane''!');
    end

    scale = 1;
    % if isBladeEffective
    %     scale = 1/3;
    % end

    Q = qf ./ (qp .* v);
    R = 1/2 * scale * Parameters.rho * pi * Parameters.R^2 * v.^2 .* ( rf - rp .* qf ./ qp );
    S = 1/2 * scale *Parameters.rho * pi * Parameters.R^2 * v.^2 .* ( sf - sp .* qf ./ qp );
end
% uuout = [Q;R;S];
end

