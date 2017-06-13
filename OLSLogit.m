
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Code for OLS estimation with Logit models. 
% 
% ARGUMENTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   J: total number of product observations
%   N: number of attributes (including price), also number of coefficients
%   K: number of "instruments" per observation
%   M: number of markets
%   Ms: M+1 vector giving indices of blocks of market data
%       E.g., Y(Ms(m):Ms(m+1),:) is attribute data in market m
% 
%   Y: J x N attribute matrix, organized in blocks according to Ms
%   s: J vector of observed shares for all products. note no share can be
%      zero; products with zero shares should be eliminated from the data
%   og: "outside good" flag ('y' or 'n', case insensitive)
%   
%   W: optional J x J symmetric positive definite weighting matrix
%   
%   beta0: initial coefficients
%   options: optimization options to use
% 
% RETURNS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%  bta: N or N+1 vector of coefficients found. N+1 if there is an outside
%        good. 
%  flag: termination flag. 
%  code: argument error codes. 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [bta,flag,code] = OLSLogit(J,N,K,M,Ms,Y,s,og,W,bta0,options,sol)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ASSERT ARGUMENT VALIDITY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % check essential argument relationships
    [flag,code] = argval(J,N,K,M,Ms,Y,s,W);
    if( flag < 0 ), 
        disp('argument error'),
        bta = []; 
        return; 
    end
    
    % ms are valid, so convert to matlab-enabled index arrays
    for m = 1:M, Msa{m} = [Ms(m):Ms(m+1)-1]; end
    
    % assert beta0 as an N-element * column * vector
    if( isempty(bta0) ), bta0 = randn(N,1); end
    sze = size( bta0 ); if( sze(2) > sze(1) ), bta0 = bta0'; end
    if( size( bta0 , 1 ) ~= N ), bta0 = randn(N,1); end
    
    % check outside good code
    switch( og ),
        case {'y','Y'},
            % must have share sums in each market less than 1
            for m = 1:M,
                if( sum( s(Msa{m}) ) == 1 ),
                    flag = -1; 
                    code = 'o';
                    return,
                end
            end
            % pad Y entries, increase N, and check initial condition
            Y = [ Y , ones(J,1) ];
            N = N + 1;
            if( size( bta0 , 1 ) < N ), bta0 = [ bta0 ; randn(1) ]; end
        otherwise, og = 'n';
    end
    
    % assert default options
    if( sol == 'k' ), % knitro
        solver = @ktrlink;
        if( isempty(options) ), 
            options = optimset('ktrlink');
            options.Display     = 'iter';
            options.GradObj     = 'on';
            options.GradConstr  = 'on';
            options.Algorithm   = 'interior-point';
        end
    else,
        solver = @fmincon;
        if( isempty(options) ), 
            options = optimset('fminunc');
            options.Display     = 'iter';
            options.GradObj     = 'on';
            options.GradConstr  = 'on';
            % options.Algorithm   = 'interior-point';
            options.Algorithm   = 'sqp';
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SOLVE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % initial condition
    x0 = [ bta0 ; randn(J,1) ];
    
    % objective
    if( isempty(W) ), obj = @(x)( obj__( N , x(N+1:N+J) ) );
    else, obj = @(x)( obj_W( N , W , x(N+1:N+J) ) );
    end
    
    % constraints
    switch( og ),
        case {'y','Y'}, 
            nlc = @(x)( residualeqn_og(J,N,M,Msa,Y,s,x(1:N),x(N+1:N+J)) );
        otherwise,
            nlc = @(x)( residualeqn(J,N,M,Msa,Y,s,x(1:N),x(N+1:N+J)) );
    end
    
    % solve
    [x,fval,flag,out] = solver(obj,x0,[],[],[],[],[],[],nlc,options);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % POST-PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    bta = x(1:N);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Determine if essential arguments are valid
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [flag,code] = argval(J,N,K,M,Ms,Y,s,W)

    flag = 1;
    code = '';

    % check that essential arrays are non-zero
    if( J <= 0 ), flag = -1; code = 'J'; return, end,
    if( N <= 0 ), flag = -1; code = 'N'; return, end,
    if( M <= 0 ), flag = -1; code = 'M'; return, end,
    if( M == 1 ), if( isempty(Ms) ), Ms = [1,J]; end
    else, 
        if( isempty(Ms) ), 
            flag = -1; 
            code = 'S'; 
            return,
        else, % Ms is not empty
            if( size( Ms , 2 ) < M+1 ),
                flag = -1; code = 'S'; return,
            end
            if( Ms(M+1) > J+1 ),
                flag = -1; code = 'S'; return,
            end
        end,
    end
    if( isempty(Y) ), flag = -1; code = 'Y'; return, end,
    if( isempty(s) ), flag = -1; code = 's'; return, end,
    
    % match sizes of arrays passed in
    sze = size(Y);
    if( sze(1) ~= J || sze(2) ~= N ), 
        flag = -1; 
        code = 'Y'; 
        return; 
    end
    
    sze = size(s);
    if( sze(1) ~= J || sze(2) ~= 1 ), 
        flag = -1; 
        code = 's'; 
        return; 
    end
    
    if( ~isempty(W) ),
        sze = size(W);
        if( sze(1) ~= J || sze(2) ~= J ), 
            flag = -1; 
            code = 'W'; 
            return; 
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% objective and constraints used above
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% with W
function [f,gf] = obj_W(N,W,res)

    gf = W * res;
    f  = res' * gf / 2;
    if( nargout > 1 ), gf = [ zeros(N,1) ; gf ]; end
    
end

% no W
function [f,gf] = obj__(N,res)

    f = res' * res / 2;
    if( nargout > 1 ), gf = [ zeros(N,1) ; res ]; end

end

function [cin,ceq,Gcin,Gceq] = residualeqn(J,N,M,Msa,Y,s,bta,res)

    cin = [];
    
    % utilities
    U = Y * bta;
    
    % safeguarded probabilities and inclusive values
    PL = zeros(J,1);
    for m = 1:M,
        umax = max( U(Msa{m}) ); 
        PL(Msa{m}) = exp( U(Msa{m}) - umax );
        SL = sum( PL(Msa{m}) );
        PL(Msa{m}) = PL(Msa{m}) / SL; 
    end
    
    % constraint value
    ceq = PL - res - s;
    
    if( nargout > 2 ), 
        
        Gcin = [];
        Gceq = zeros(N+J,J);
        
        Gceq(1:N,:) = Y';
        for m = 1:M, 
            GGm = Y(Msa{m},:)' * PL(Msa{m});
            for j = Msa{m},
                Gceq(1:N,j) = PL(j) * ( Gceq(1:N,j) - GGm );
                Gceq(N+j,j) = - 1.0; 
            end
        end
        
    end

end

function [cin,ceq,Gcin,Gceq] = residualeqn_og(J,N,M,Msa,Y,s,bta,res)

    cin = [];
    
    % utilities
    U = Y * bta;
    
    % safeguarded probabilities and inclusive values
    PL = zeros(J,1);
    for m = 1:M,
        umax = max( U(Msa{m}) ); 
        PL(Msa{m}) = exp( U(Msa{m}) - umax );
        SL = exp( -umax ) + sum( PL(Msa{m}) );
        PL(Msa{m}) = PL(Msa{m}) / SL; 
    end
    
    % constraint value
    ceq = PL - res - s;
    
    if( nargout > 2 ), 
        
        Gcin = [];
        Gceq = zeros(N+J,J);
        
        Gceq(1:N,:) = Y';
        for m = 1:M, 
            GGm = Y(Msa{m},:)' * PL(Msa{m});
            for j = Msa{m},
                Gceq(1:N,j) = PL(j) * ( Gceq(1:N,j) - GGm );
                Gceq(N+j,j) = - 1.0; 
            end
        end
        
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%