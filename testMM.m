% test script

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

J  = 500;   % number of products
N  = 2;     % number of attributes

M  = 3;     % number of markets
Ms = [1,101,201,J+1]; % market blocks 

% attributes (random)
Y  = randn(J,N);

% "true" betas (in case we use outside good)
btaT = rand(N+1,1);

% outside good
og = 'y';

% fmincon or ktrlink?
sol = 'f';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% shares (Logit choice probabilities)
switch( og )
    
    case {'y','Y'}
        U = [ Y , ones(J,1) ] * btaT;
        s = zeros(J,1);
        for m = 1:M,
            s(Ms(m):Ms(m+1)-1) = exp( U(Ms(m):Ms(m+1)-1) );
            SL = 1 + sum( s(Ms(m):Ms(m+1)-1) );
            s(Ms(m):Ms(m+1)-1) = s(Ms(m):Ms(m+1)-1) / SL;
        end
        
    otherwise,
        U = Y * btaT(1:N);
        s = zeros(J,1);
        for m = 1:M,
            s(Ms(m):Ms(m+1)-1) = exp( U(Ms(m):Ms(m+1)-1) );
            SL = sum( s(Ms(m):Ms(m+1)-1) );
            s(Ms(m):Ms(m+1)-1) = s(Ms(m):Ms(m+1)-1) / SL;
        end
        
end

% initial condition
bta0 = [];

% options and solver
switch( sol ), 
    
    case 'k',
        opt = optimset('ktrlink');
        opt.Display     = 'iter';
        opt.GradObj     = 'on';
        opt.GradConstr  = 'on';
        opt.Algorithm   = 'interior-point';
        % opt.DerivativeCheck = 'on';
        
    otherwise,
        opt = optimset('fmincon');
        opt.Display     = 'iter';
        opt.GradObj     = 'on';
        opt.GradConstr  = 'on';
        % opt.Algorithm   = 'interior-point';
        opt.Algorithm   = 'sqp';
        % opt.DerivativeCheck = 'on';
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SOLUTION TESTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% "nothing"

% instruments
K = 0; Z = [];

% no weighting
w = 0; W = [];

% solving
[bta,flag,code] = GMMLogit(J,N,K,M,Ms,Y,s,og,Z,w,W,bta0,opt,sol);

flag, if( flag < 0 ), code, end
bta, btaT,

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generic weighting

% instruments
K = 0; Z = [];

% generic weighting
w = 2; R = triu(randn(J,J)); W = R' * R;

% solving
[bta,flag,code] = GMMLogit(J,N,K,M,Ms,Y,s,og,Z,w,W,bta0,opt,sol);

flag, if( flag < 0 ), code, end
bta, btaT,

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% instrumented, not weighted

% instruments
K = 1; Z = randn(J,K);

% no weighting
w = 0; W = [];

% solving
[bta,flag,code] = GMMLogit(J,N,K,M,Ms,Y,s,og,Z,w,W,bta0,opt,sol);

flag, if( flag < 0 ), code, end
bta, btaT,

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% instrumented, Z weighting

% instruments
K = 1; Z = randn(J,K);

% Z weighting
w = 1; W = [];

% solving
[bta,flag,code] = GMMLogit(J,N,K,M,Ms,Y,s,og,Z,w,W,bta0,opt,sol);

flag, if( flag < 0 ), code, end
bta, btaT,

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% instrumented, generic weighting

% instruments
K = 2; Z = randn(J,K);

% generic weighting
w = 2; R = triu(randn(K,K)); W = R'*R;

% solving
[bta,flag,code] = GMMLogit(J,N,K,M,Ms,Y,s,og,Z,w,W,bta0,opt,sol);

flag, if( flag < 0 ), code, end
bta, btaT,

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[bta,flag,code] = MLELogit(J,N,M,Ms,Y,s,og,bta0,opt,sol);

flag, if( flag < 0 ), code, end
bta, btaT,

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[bta,flag,code] = OLSLogit(J,N,K,M,Ms,Y,s,og,[],bta0,opt,sol);

flag, if( flag < 0 ), code, end
bta, btaT,

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
