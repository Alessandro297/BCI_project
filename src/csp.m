function W = csp(X1, X2)  
    %W = mixing matrix X1 = data from first class | X2 = data from second class
       
    % compute covariance matrix of each class
    S1 = cov(X1');  
    S2 = cov(X2');   
    % Solve the eigenvalue problem S1·W = l·S2·W
    [W,L] = eig(S1, S1 + S2);   % Mixing matrix W (spatial filters are columns)
end

