
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Code for BLP-style GMM estimation with Logit models. 
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
%   Y: J x K attribute matrix, organized in blocks according to Ms
%   s: J vector of observed shares for all products. note no share can be
%      zero; products with zero shares should be eliminated from the data
%   og: "outside good" flag ('y' or 'n', case insensitive)
%   
%   Z: N x L "instrument" matrix (none used if null)
%   w: weighting code (0 = none, 1 = inv( Z' Z ), 2 = generic)
%   W: L x L symmetric positive definite weighting matrix (if w = 2 only)
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

function [bta,eta,flag,code] ...
                = TSRLogit(J,...        number of products
                           N,...        number of data observations
                           K,...        number of characteristics per product
                           L,...        number of instruments (if used)
                           M,...        number of "markets"
                           Ms,...       M-element cell array listing indices of products in each market
                           Y,...        J x K matrix of product characteristics
                           s,...        N-element vector of shares in each market
                           og,...       outside good flag
                           Z,...        N x L matrix of instruments (optional)
                           w,...        weighting code
                           W,...        L x L or N x N weighting matrix (if generic)
                           bta0,...     initial condition on beta's
                           eta0,...     initial condition on eta (autocorelation parameter)
                           options,...  options for solver
                           sol)       % solver to use
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ASSERT ARGUMENT VALIDITY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % check essential argument relationships
    [flag,code] = argval(J,N,K,M,Ms,Y,s,Z,W);
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
    
    % check GMM weighting code
    switch( w ),
        case 1, if( isempty(Z) ), w = 0; end % can't use inv( Z' Z ) w/o Z
        case 2, if( isempty(W) ), w = 0; end % can't use W w/o W
        otherwise, w = 0;
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
    
    if( isempty(Z) ), % no instruments
        [bta,flag,out] ...
            = GMMLogit_noZ(J,N,K,M,Msa,Y,s,og,w,W,bta0,options,solver);
    else,  % using instruments
        switch( w ),
            % inv( Z' Z ) weighting
            case 1, 
                [bta,flag,out] ...
                    = GMMLogit_w1Z(J,N,K,M,Msa,Y,s,og,Z,bta0,options,solver);
            % generic weighting
            case 2, 
                [bta,flag,out] ...
                    = GMMLogit_w2Z(J,N,K,M,Msa,Y,s,og,Z,W,bta0,options,solver);
            % no weighting (identity)
            otherwise,
                [bta,flag,out] ...
                    = GMMLogit_w0Z(J,N,K,M,Msa,Y,s,og,Z,bta0,options,solver);
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % POST-PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
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
function [flag,code] = argval(J,N,K,M,Ms,Y,s,Z,W)

    flag = 1;
    code = '';

    % check that essential arrays are non-zero
    if( J <= 0 ), flag = -1; code = 'J'; return, end,
    if( N <= 0 ), flag = -1; code = 'N'; return, end,
    if( M <= 0 ), flag = -1; code = 'M'; return, end,
    if( M == 1 ), if( isempty(Ms) ), Ms = [1,J+1]; end,
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
    
    if( ~isempty(Z) ),
        sze = size(Z);
        if( sze(1) ~= J || sze(2) ~= K ), 
            flag = -1; 
            code = 'Z'; 
            return; 
        end
    end
    
    if( ~isempty(W) ),
        if( isempty(Z) ), 
            sze = size(W);
            if( sze(1) ~= J || sze(2) ~= J ), 
                flag = -1; 
                code = 'W'; 
                return; 
            end
        else,
            sze = size(W);
            if( sze(1) ~= K || sze(2) ~= K ), 
                flag = -1; 
                code = 'W'; 
                return; 
            end
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Z is included, but no weighting matrix. Solve
% 
%       min g' g / 2
%       wrt beta, x, g
%       sto - Z' x + g = 0                  (linear equality constraint)
%           log( P(beta,x) ) - log( s ) = 0 (nonlinear equality constraint)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [bta,flag,out] ...
                = GMMLogit_w0Z(J,N,K,M,Msa,Y,s,og,Z,bta0,options,solver)

    % objective
    obj = @(x)( objZ_(J,N,x(N+J+1:N+J+K)) ); % g's are last K variables
    
    % initial condition
    x0 = [ bta0 ; randn(J,1) ; randn(K,1) ]; % beta, then xsi, then g
    
    % linear constraints
    Aeq = [ zeros(K,N) , - Z' , eye(K) ]; 
    beq = zeros(K,1);
    
    % nonlinear constraints
    % (have to pad K x J zero matrix on bottom of derivatives)
    switch( og ),
        case {'y','Y'},
            nlc = @(x)( shreqncns_og(J,N,K,M,Msa,Y,s,og,x(1:N),x(N+1:N+J)) );
        otherwise,
            nlc = @(x)( shreqncns(J,N,K,M,Msa,Y,s,og,x(1:N),x(N+1:N+J)) );
    end
    
    % solve
    [x,fval,flag,out] = solver(obj,x0,[],[],Aeq,beq,[],[],nlc,options);
    
    % assign coefficients for return
    bta = x(1:N);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Z is included, and weighting matrix is inv( Z' Z ). Solve
% 
%       min g' g / 2
%       wrt beta, x, g
%       sto - Z' x + V D g = 0              (linear equality constraint)
%           log( P(beta,x) ) - log( s ) = 0 (nonlinear equality constraint)
% 
% where Z = U [ D ; 0 ] V' is an svd. 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [bta,flag,out] ...
                = GMMLogit_w1Z(J,N,K,M,Msa,Y,s,og,Z,bta0,options,solver)

    % objective
    obj = @(x)( objZ_(J,N,x(N+J+1:N+J+K)) ); % g's are last K variables
    
    % initial condition
    x0 = [ bta0 ; randn(J,1) ; randn(K,1) ]; % beta, then xsi, then g
    
    % linear constraints
    [U,D,V] = svd( Z );
    if( K == 1 ), d = D(1,1);
    else, d = diag(D);
    end
    Aeq = [ zeros(K,N) , - Z' , V * diag(d) ]; 
    beq = zeros(K,1);
    
    clear U D d V,
    
    % nonlinear constraints
    % (have to pad K x J zero matrix on bottom of derivatives)
    switch( og ),
        case {'y','Y'},
            nlc = @(x)( shreqncns_og(J,N,K,M,Msa,Y,s,og,x(1:N),x(N+1:N+J)) );
        otherwise,
            nlc = @(x)( shreqncns(J,N,K,M,Msa,Y,s,og,x(1:N),x(N+1:N+J)) );
    end
    
    % solve
    [x,fval,flag,out] = solver(obj,x0,[],[],Aeq,beq,[],[],nlc,options);

    % assign coefficients for return
    bta = x(1:N);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Z is included, and weighting matrix is given. Solve
% 
%       min g' g / 2
%       wrt beta, x, g
%       sto - R Z' x + g = 0                (linear equality constraint)
%           log( P(beta,x) ) - log( s ) = 0 (nonlinear equality constraint)
% 
% where R is W's Cholesky factor (W = R' R, R upper triangular)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [bta,flag,out] ...
                = GMMLogit_w2Z(J,N,K,M,Msa,Y,s,og,Z,W,bta0,options,solver)

    % objective
    obj = @(x)( objZ_(J,N,x(N+J+1:N+J+K)) ); % g's are last K variables
    
    % initial condition
    x0 = [ bta0 ; randn(J,1) ; randn(K,1) ]; % beta, then xsi, then g
    
    % linear constraints
    R = chol(W); % uses only upper triangle of W
    Aeq = [ zeros(K,N) , - R * Z' , eye(K) ]; 
    beq = zeros(K,1);
    
    clear R,
    
    % nonlinear constraints
    % (have to pad K x J zero matrix on bottom of derivatives)
    switch( og ),
        case {'y','Y'},
            nlc = @(x)( shreqncns_og(J,N,K,M,Msa,Y,s,og,x(1:N),x(N+1:N+J)) );
        otherwise,
            nlc = @(x)( shreqncns(J,N,K,M,Msa,Y,s,og,x(1:N),x(N+1:N+J)) );
    end
    
    % solve
    [x,fval,flag,out] = solver(obj,x0,[],[],Aeq,beq,[],[],nlc,options);

    % assign coefficients for return
    bta = x(1:N);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Z is not included. Solve
% 
%       min res' W res / 2
%       wrt beta, eta, xsi, lag, res
%       sto log( P(beta,xsi) ) - log( s ) = 0   for all n
%           lag(n) - (-eta)^(t(n)) = 0          for all n
%           res(n) - sum_{s=1}^{T(j(n))} lag(n(j(n),s)) xsi(n(j(n),t-s)) = 0 for all n
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [bta,flag,out] ...
                = GMMLogit_noZ(J,N,K,M,Msa,Y,s,og,w,W,bta0,options,solver)
    
    % objective
    switch( w ),
        case '2',  obj = @(x)( obj_W( N , W , x(N+1:N+J) ) ); 
        otherwise, obj = @(x)( obj__( N ,     x(N+1:N+J) ) );
    end

    % initial condition: 
    % variable order is beta, eta, xsi, lagged terms, residuals
    x0 = [ bta0 ; ...       utility coefficients (beta)
           eta0 ; ...       autocorrelation parameter (eta)
           randn(N,1) ; ... xsi terms: P(beta,xsi) = s
           randn(N,1) ; ... "lagged" terms: N(n) - (-eta)^(t(n)) = 0
           randn(N,1) ]; %  residual terms: r(n(j,t)) = sum_{s} N(n(j,s)) xsi(n(j,t-s)) for all j,t

    % nonlinear constraints
    switch( og ),
        case {'y','Y'},
            nlc = @(x)( shreqncns_og(J,N,0,M,Msa,Y,s,og,x(1:N),x(N+1:N+J)) );
        otherwise,
            nlc = @(x)( shreqncns(J,N,0,M,Msa,Y,s,og,x(1:N),x(N+1:N+J)) );
    end

    % solve
    [x,fval,flag,out] = solver(obj,x0,[],[],[],[],[],[],nlc,options);

    % assign coefficients for return
    bta = x(1:K);
    eta = x(K+1);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% objectives used above
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% with Z, with or without W
function [f,gf] = objZ_(J,N,g)

    f = g' * g / 2;
    if( nargout > 1 ), 
        gf = [ zeros(N,1) ; zeros(J,1) ; g ];
    end

end

% no Z, with W
function [f,gf] = obj_W(K,N,W,res)

    gf = W * res;
    f  = res' * gf / 2;
    
    % have to pad zeros on derivative
    if( nargout > 1 ), gf = [ zeros(K+1+N,1) ; gf ]; end,
    
end

% no Z, no W
function [f,gf] = obj__(K,N,xsi)

    f = res' * res / 2;
    
    % have to pad zeros on derivative
    if( nargout > 1 ), gf = [ zeros(K+1+N,1) ; gf ]; end,

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% share equation constraints
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% no outside good
function [cin,ceq,Gcin,Gceq] = shreqncns(J,N,K,M,Msa,Y,s,og,bta,xsi)

    cin = [];
    ceq = zeros(J,1);

    % "allocations"
    inclv = zeros(1,M);
    PL    = zeros(J,1);
    
    % L(j) = log( P(j) ) - log( s(j) ) 
    %      = U(j) + xsi(j) - inclv( m(j) )
    %
    % where 
    % 
    %       inclv(m) = log( sum( exp( Y(m)' beta ) ) )
    %
    
    % utilities (Y a J x N matrix, beta an N x 1 vector)
    U = Y * bta + xsi; % utilities a column vector
    
    % safeguarded probabilities and inclusive values
    for m = 1:M,
        umax = max( U(Msa{m}) ); 
        PL(Msa{m}) = exp( U(Msa{m}) - umax );
        SL = sum( PL(Msa{m}) );
        PL(Msa{m}) = PL(Msa{m}) / SL;
        inclv(m) = umax + log( SL );
    end
    
    % share equation constraint value
    ceq = U - log( s );
    for m = 1:M, 
        ceq(Msa{m}) = ceq(Msa{m}) - inclv(m); 
    end
    
    % derivatives
    if( nargout > 2 ), 
        
        Gcin = [];
        
        Gceq = zeros(N+J+K,J);
        
        % beta derivatives of the share equations
        Gceq(1:N,:) = Y';
        for m = 1:M, 
            GGm = Y(Msa{m},:)' * PL(Msa{m});
            for j = Msa{m},
                Gceq(1:N,j) = Gceq(1:N,j) - GGm;
            end
        end
        
        % xsi derivatives of the share equations
        Gceq(N+1:N+J,:) = eye(J,J);
        for m = 1:M,
            for j = Msa{m},
                Gceq(N+Msa{m},j) = Gceq(N+Msa{m},j) - PL(Msa{m});
            end
        end
        
    end

end

% with outside good
function [cin,ceq,Gcin,Gceq] = shreqncns_og(J,N,K,M,Msa,Y,s,og,bta,xsi)

    cin = [];
    ceq = zeros(J,1);

    % "allocations"
    inclv = zeros(1,M);
    PL    = zeros(J,1);
    
    % L(j) = log( P(j) ) - log( s(j) ) 
    %      = U(j) + xsi(j) - inclv( m(j) )
    %
    % where 
    % 
    %       inclv(m) = log( sum( exp( Y(m)' beta ) ) )
    %
    
    % utilities (Y a J x N matrix, beta an N x 1 vector)
    U = Y * bta + xsi; % utilities a column vector
    
    % safeguarded probabilities and inclusive values
    for m = 1:M,
        umax = max( U(Msa{m}) ); 
        PL(Msa{m}) = exp( U(Msa{m}) - umax );
        SL = exp( -umax ) + sum( PL(Msa{m}) );
        PL(Msa{m}) = PL(Msa{m}) / SL;
        inclv(m) = umax + log( SL );
    end
    
    % share equation constraint value
    ceq = U - log( s );
    for m = 1:M, 
        ceq(Msa{m}) = ceq(Msa{m}) - inclv(m); 
    end
    
    % derivatives
    if( nargout > 2 ), 
        
        Gcin = [];
        
        Gceq = zeros(N+J+K,J);
        
        % beta derivatives of the share equations
        Gceq(1:N,:) = Y';
        for m = 1:M, 
            GGm = Y(Msa{m},:)' * PL(Msa{m});
            for j = Msa{m},
                Gceq(1:N,j) = Gceq(1:N,j) - GGm;
            end
        end
        
        % xsi derivatives of the share equations
        Gceq(N+1:N+J,:) = eye(J,J);
        for m = 1:M,
            for j = Msa{m},
                Gceq(N+Msa{m},j) = Gceq(N+Msa{m},j) - PL(Msa{m});
            end
        end
        
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%