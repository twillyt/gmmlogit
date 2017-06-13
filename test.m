% test script

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

J  = 10;    % number of products
N  = 2;     % number of attributes

M  = 1;     % number of markets
Ms = [1,J+1]; % market blocks 

I  = 1000;   % number of "individuals" for sample-variance draws

% attributes (random)
Y  = randn(J,N);

% "true" betas (in the case where we use outside good)
btaT = rand(N,1); btaT = [ btaT ; - 0.9 * max(btaT) ];

% outside good
og = 'y';

% fmincon or ktrlink?
sol = 'k';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initial condition
bta0 = [];

% options and solver
switch( sol ), 
    
    case 'k',
        
        opt = optimset('ktrlink');
        opt.Display     = 'final';
        opt.GradObj     = 'on';
        opt.GradConstr  = 'on';
        opt.Algorithm   = 'interior-point';
        % opt.DerivativeCheck = 'on';
        
    otherwise,
        opt = optimset('fmincon');
        opt.Display     = 'final';
        opt.GradObj     = 'on';
        opt.GradConstr  = 'on';
        opt.Algorithm   = 'interior-point';
        % opt.Algorithm   = 'sqp';
        % opt.DerivativeCheck = 'on';
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PPV/NPV TESTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% draw shares
s = drawshares( Y , btaT , J , M , Ms , 100 , og );

% compute model estimates
[bta,flag,code] = MLELogit(J,N,M,Ms,Y,s,og,bta0,opt,sol);

% now, approximate PPV / NPV; i.e., probability that a product is chosen
% given that the model says it is chosen. This is a function of x... so
% first define x
x = randn(N,1);
% now, draw choices with each model (do we equate errors?)
for i = 1:1000,
    
    
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SOLUTION TESTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



Is = [10,100,1000,10000,100000];

figure(1), clf, 
for n = 1:N,
    subplot(1,N,n), 
    semilogx( [min(Is),max(Is)] , [0,0] , '--k' ),
end

for t = 1:size(Is,2),
    
    for T = 1:10,
        s = drawshares( Y , btaT , J , M , Ms , Is(t) , og );
        [bta,flag,code] = MLELogit(J,N,M,Ms,Y,s,og,bta0,opt,sol);
        for n = 1:N,
            subplot(1,N,n), hold on, 
            semilogx( Is(t) , 100*(bta(n)-btaT(n))/abs(btaT(n)) , ...
                        '.k' , 'MarkerSize' , 20 ),
        end
    end
    
    s = drawshares( Y , btaT , J , M , Ms , Is(t) , og );
    [bta,flag,code] = MLELogit(J,N,M,Ms,Y,s,og,bta0,opt,sol);
    
    % assume this last model is accurate
    for T = 1:10,
        s1 = drawshares( Y , bta , J , M , Ms , I , og );
        [bta1,flag,code] = MLELogit(J,N,M,Ms,Y,s1,og,bta,opt,sol);
        for n = 1:N,
            subplot(1,N,n), hold on, 
            semilogx( Is(t) , 100*(bta1(n)-bta(n))/abs(bta(n)) , ...
                        '.r' , 'MarkerSize' , 20 ),
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MLE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% solve with MLE
[bta,flag,code] = MLELogit(J,N,M,Ms,Y,s,og,bta0,opt,sol);

flag, if( flag < 0 ), code, end
bta, btaT,

% take coefficients at face value, and resample to determine a coefficient
% range
btaR = [ bta , bta ];
for t = 1:10,
    
    % draw shares assuming the model made is accurate
    s1 = drawshares( Y , bta , J , M , Ms , I , og );
    
    % solve for MLE estimates
    [btaS(:,t),flag,code] = MLELogit(J,N,M,Ms,Y,s1,og,bta,opt,sol);
    
    btaR(:,1) = min( btaR(:,1) , btaS(:,t) );
    btaR(:,2) = max( btaR(:,2) , btaS(:,t) );
    
end

btaS, btaR, btaT

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % no instruments, 
% 
% % instruments
% K = 0; Z = [];
% 
% % no weighting
% w = 0; W = [];
% 
% % solving
% [bta,flag,code] = GMMLogit(J,N,K,M,Ms,Y,s,og,Z,w,W,bta0,opt,sol);
% 
% flag, if( flag < 0 ), code, end
% bta, btaT,
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % generic weighting
% 
% % instruments
% K = 0; Z = [];
% 
% % generic weighting
% w = 2; R = triu(randn(J,J)); W = R' * R;
% 
% % solving
% [bta,flag,code] = GMMLogit(J,N,K,M,Ms,Y,s,og,Z,w,W,bta0,opt,sol);
% 
% flag, if( flag < 0 ), code, end
% bta, btaT,
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % instrumented, not weighted
% 
% % instruments
% K = 1; Z = randn(J,K);
% 
% % no weighting
% w = 0; W = [];
% 
% % solving
% [bta,flag,code] = GMMLogit(J,N,K,M,Ms,Y,s,og,Z,w,W,bta0,opt,sol);
% 
% flag, if( flag < 0 ), code, end
% bta, btaT,
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % instrumented, Z weighting
% 
% % instruments
% K = 1; Z = randn(J,K);
% 
% % Z weighting
% w = 1; W = [];
% 
% % solving
% [bta,flag,code] = GMMLogit(J,N,K,M,Ms,Y,s,og,Z,w,W,bta0,opt,sol);
% 
% flag, if( flag < 0 ), code, end
% bta, btaT,
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % instrumented, generic weighting
% 
% % instruments
% K = 2; Z = randn(J,K);
% 
% % generic weighting
% w = 2; R = triu(randn(K,K)); W = R'*R;
% 
% % solving
% [bta,flag,code] = GMMLogit(J,N,K,M,Ms,Y,s,og,Z,w,W,bta0,opt,sol);
% 
% flag, if( flag < 0 ), code, end
% bta, btaT,

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% [bta,flag,code] = OLSLogit(J,N,K,M,Ms,Y,s,og,[],bta0,opt,sol);
% 
% flag, if( flag < 0 ), code, end
% bta, btaT,

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
