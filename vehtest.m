% vehicle test

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load x1.mat

Y = x1; clear x1,

[J,N] = size(Y); % J ~ 739

M  = 3;     % number of markets (panel years)
Ms = [1,248,499,J+1]; % market blocks (from statdata.xls)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load sales.mat

% outside good
og = 'n';
switch( og ), 
    case {'y','Y'}, ogsales = 100000;
    otherwise,      ogsales = 0;
end

% shares (market data)
s = zeros(J,1);
for m = 1:M,
    s(Ms(m):Ms(m+1)-1) = sales(Ms(m):Ms(m+1)-1);
    SS = ogsales + sum( s(Ms(m):Ms(m+1)-1) );
    s(Ms(m):Ms(m+1)-1) = s(Ms(m):Ms(m+1)-1)/ SS;
end

clear sales,

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load z.mat

Z = z; clear z,

K = size( Z ,2 );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
%
% [bta,flag,code] = solver(J,N,K,M,Ms,Y,s,og,Z,w,W,bta0,options)
%

% testcase = 4;

switch( testcase ),
    
    case 1, % no instruments, no weighting

        [bta,flag,code] = GMMLogit(J,N,0,M,Ms,Y,s,og,[],0,[],bta0,opt,sol);

        flag, if( flag < 0 ), code, end
        bta,
        
    case 2, % no instruments, random weighting

        % random weighting
        R = triu(randn(J,J)); W = R' * R;

        % solving
        [bta,flag,code] = GMMLogit(J,N,0,M,Ms,Y,s,og,[],2,W,bta0,opt,sol);

        flag, if( flag < 0 ), code, end
        bta,

    case 3, % instrumented, not weighted

        % solving
        [bta,flag,code] = GMMLogit(J,N,K,M,Ms,Y,s,og,Z,0,[],bta0,opt,sol);

        flag, if( flag < 0 ), code, end
        bta,
        
    case 4, % instrumented, Z weighting

        % solving
        [bta,flag,code] = GMMLogit(J,N,K,M,Ms,Y,s,og,Z,1,[],bta0,opt,sol);

        flag, if( flag < 0 ), code, end
        bta,

    case 5, % instrumented, random weighting

        % generic weighting
        R = triu(randn(K,K)); W = R'*R;

        % solving
        [bta,flag,code] = GMMLogit(J,N,K,M,Ms,Y,s,og,Z,2,W,bta0,opt,sol);

        flag, if( flag < 0 ), code, end
        bta,
        
    otherwise,

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[bta,flag,code] = MLELogit(J,N,M,Ms,Y,s,og,bta0,opt,sol);

flag, if( flag < 0 ), code, end
bta,

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[bta,flag,code] = OLSLogit(J,N,K,M,Ms,Y,s,og,[],bta0,opt,sol);

flag, if( flag < 0 ), code, end
bta,
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
