
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Code for MLE estimation of Logit models. 
% 
% DISCUSSION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% ARGUMENTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   J: total number of product observations
%   N: number of attributes (including price), also number of coefficients
%   M: number of markets
%   Ms: M+1 vector giving indices of blocks of market data
%       E.g., Y(Ms(m):Ms(m+1),:) is attribute data in market m
% 
%   Y: J x N attribute matrix, organized in blocks according to Ms
%   s: J vector of observed shares for all products. note no share can be
%      zero; products with zero shares should be eliminated from the data
%   og: "outside good" flag ('y' or 'n', case insensitive)
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
% NOTES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [bta,flag,code] = MLELogit(J,N,M,Ms,Y,s,og,bta0,options,sol)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ASSERT ARGUMENT VALIDITY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % check essential argument relationships
    [flag,code] = argval(J,N,M,Ms,Y,s);
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
            % must have share sums in each market * less than * 1
%             for m = 1:M,
%                 if( sum( s(Msa{m}) ) == 1 ),
%                     flag = -1; 
%                     code = 'o';
%                     return,
%                 end
%             end
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
    x0 = [ bta0 ; randn(M,1) ];
    
    % objective
    v = Y' * s;
    obj = @(x)( negloglik(M,v,x(1:N),x(N+1:N+M)) );

    % constraints
    switch( og ),
        case {'y','Y'}, 
            nlc = @(x)( inclvcons_og(J,N,M,Msa,Y,x(1:N),x(N+1:N+M)) );
        otherwise,
            nlc = @(x)( inclvcons(J,N,M,Msa,Y,x(1:N),x(N+1:N+M)) );
    end
    
    % solve
    [x,fval,flag,out] = solver(obj,x0,[],[],[],[],[],[],nlc,options);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % POST-PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    bta = x(1:N);
    
    % switch( og ),
    %     case {'y','Y'}, bta = x(1:N+1);
    %     otherwise, bta = x(1:N);
    % end
    
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
function [flag,code] = argval(J,N,M,Ms,Y,s)

    flag = 1;
    code = '';

    % check that essential arrays are non-zero
    if( J <= 0 ), flag = -1; code = 'J'; return, end,
    if( N <= 0 ), flag = -1; code = 'N'; return, end,
    if( M <= 0 ), flag = -1; code = 'M'; return, end,
    if( M == 1 ), if( isempty(Ms) ), Ms = [1,J+1]; end
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
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% negative log likelihood function
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [nll,Dnll] = negloglik(M,v,bta,inclv)

    % ll(j) = sum_m sum_j s(m,j) log( P(m,j) )
    %       = sum_m sum_j s(m,j) ( Y({j,m},:) bta - inclv(m) )
    %       = s' Y bta - sum_m ( sum_{j in J(m)} s(j) ) inclv(m)
    % 
    % where
    % 
    %       inclv(m) = log( sum_{j in J(m)} exp( Y(j,:) beta ) )
    % 
    % Note, however, that 
    % 
    %       sum_{j in J(m)} s(j) = 1
    % 
    % (by definition) and thus
    % 
    %       ll(j) = s' Y bta - sum_m inclv(m)
    % 
    % we can thus view the ll function as a linear function in beta and
    % inclusive values
    %
    %       [ Y' s ]' [  bta  ]
    %       [  -1  ]  [ inclv ]
    %       
    % and constrain the inclusive values
    % 
    %       log( ... ) - inclv(m) = 0
    % 
    % When there is an outside good, 
    % 
    % ll(j) = sum_m ( s(m,0) log( P(m,0) ) + sum_j s(m,j) log( P(m,j) ) )
    %       = sum_m ( - s(m,0) inclv(m) 
    %                   + sum_j s(m,j) ( Y({j,m},:) bta - inclv(m) ) )
    %       = s' Y bta - sum_m ( s(m,0) + sum_{j in J(m)} s(j) ) inclv(m)
    %       = s' Y bta - sum_m inclv(m)
    % 
    % where
    % 
    %       log( 1 + sum_{j in J(m)} exp( Y(j,:) beta ) ) - inclv(m) = 0
    % 
    % is the constraint. 
    
    nll  = - v' * bta + sum( inclv );
    Dnll = [ - v ; ones(M,1) ];
    
end

function [cin,ceq,Gcin,Gceq] = inclvcons(J,N,M,Msa,Y,bta,inclv)

    cin = [];
    ceq = zeros(M,1);
    
    % utilities
    U = Y * bta;
    
    % safeguarded probabilities and inclusive values
    PL = zeros(J,1);
    for m = 1:M,
        umax = max( U(Msa{m}) ); 
        PL(Msa{m}) = exp( U(Msa{m}) - umax );
        SL = sum( PL(Msa{m}) );
        PL(Msa{m}) = PL(Msa{m}) / SL;
        ceq(m) = umax + log( SL ) - inclv(m);
    end
    
    if( nargout > 2 ), 
        
        Gcin = [];
        Gceq = zeros(N+M,M);
        
        for m = 1:M, 
            Gceq(1:N,m) = Y(Msa{m},:)' * PL(Msa{m});
            Gceq(N+m,m) = - 1.0;
        end
        
    end

end

function [cin,ceq,Gcin,Gceq] = inclvcons_og(J,N,M,Msa,Y,bta,inclv)

    cin = [];
    ceq = zeros(M,1);
    
    % utilities
    U = Y * bta;
    
    % safeguarded probabilities and inclusive values
    PL = zeros(J,1);
    for m = 1:M,
        umax = max( U(Msa{m}) ); 
        PL(Msa{m}) = exp( U(Msa{m}) - umax );
        SL = exp( -umax ) + sum( PL(Msa{m}) );
        PL(Msa{m}) = PL(Msa{m}) / SL;
        ceq(m) = umax + log( SL ) - inclv(m);
    end
    
    if( nargout > 2 ), 
        
        Gcin = [];
        Gceq = zeros(N+M,M);
        
        for m = 1:M,
            Gceq(1:N,m) = Y(Msa{m},:)' * PL(Msa{m});
            Gceq(N+m,m) = - 1.0;
        end
        
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%