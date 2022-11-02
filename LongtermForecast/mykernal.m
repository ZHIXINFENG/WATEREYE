function KMN = mykernal(XM,XN,theta)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
N = size(XM,2);
param = exp(theta);
sigmaL1 = params(1:D,1);
sigmaL2 = params(D+1,1);
sigmaF1 = params(D+2,1);
sigmaF2 = params(D+3,1);

% 3. Create the contribution due to squared exponential ARD.    
KMN = pdist2(XM(:,1)/sigmaL1(1), XN(:,1)/sigmaL1(1)).^2;
for r = 2:D
  KMN = KMN + pdist2(XM(:,r)/sigmaL1(r), XN(:,r)/sigmaL1(r)).^2;        
end
KMN = (sigmaF1^2)*exp(-0.5*KMN);
% 4. Add the contribution due to squared exponential.
KMN = KMN + (sigmaF2^2)*exp(-0.5*(pdist2(XM/sigmaL2, XN/sigmaL2).^2)); 
end

