
function s = drawshares( Y , bta , J , M , Ms , I , og );

    % shares (Logit choice probabilities). 
    switch( og )

        case {'y','Y'}

            U = [ Y , ones(J,1) ] * bta;
            s = zeros(J,1);
            
            if( I > 0 || I < Inf ),

                % This method of sampling includes sample variance
                for m = 1:M,
                    Jm = Ms(m+1) - Ms(m); % number of products in this market
                    for i = 1:I,
                        ue = - 0.5772156649 - log( - log(rand(Jm,1)) ); % errors
                        ui = U(Ms(m):Ms(m+1)-1) + ue; % utilities, with errors
                        j = 0; uim = 0; % og utility is zero
                        for k = 1:Jm,
                            if( ui(k) > uim ), j = k; uim = ui(k); end
                        end
                        if( j > 0 ), s(Ms(m)+j-1) = s(Ms(m)+j-1) + 1; end
                    end
                    s(Ms(m):Ms(m+1)-1) = s(Ms(m):Ms(m+1)-1) / I;
                end

            else, 

                % This method of drawing shares ignores sample variance
                for m = 1:M,
                    s(Ms(m):Ms(m+1)-1) = exp( U(Ms(m):Ms(m+1)-1) );
                    SL = 1 + sum( s(Ms(m):Ms(m+1)-1) );
                    s(Ms(m):Ms(m+1)-1) = s(Ms(m):Ms(m+1)-1) / SL;
                end
            
            end

        otherwise,

            U = Y * bta(1:N);
            s = zeros(J,1);
            
            if( I > 0 || I < Inf ),

                % This method of sampling includes sample variance
                for m = 1:M,
                    Jm = Ms(m+1) - Ms(m); % number of products in this market
                    for i = 1:I,
                        ue = - 0.5772156649 - log( - log(rand(Jm,1)) ); % errors
                        ui = U(Ms(m):Ms(m+1)-1) + ue; % utilities, with errors
                        j = 1; uim = ui(1);
                        for k = 2:Jm,
                            if( ui(k) > uim ), j = k; uim = ui(k); end
                        end
                        s(Ms(m)+j-1) = s(Ms(m)+j-1) + 1;
                    end
                    s(Ms(m):Ms(m+1)-1) = s(Ms(m):Ms(m+1)-1) / I;
                end
                
            else, 

                % This method of drawing shares ignores sample variance
                for m = 1:M,
                    s(Ms(m):Ms(m+1)-1) = exp( U(Ms(m):Ms(m+1)-1) );
                    SL = sum( s(Ms(m):Ms(m+1)-1) );
                    s(Ms(m):Ms(m+1)-1) = s(Ms(m):Ms(m+1)-1) / SL;
                end
            
            end

    end

end