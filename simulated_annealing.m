f = @(x) sinc(x);                            % the objective function. 
LeftBound = -3; 
RightBound = 3;
N = 10000;                                    % Number of sample points. 
xs = linspace(LeftBound, RightBound, N);         % the log space. 
Ts = linspace(10, 1e-2, 5);                   % temperature. 
hold on; 
for T = Ts
    g = @(x) log(exp(-f(x)/T) + 1);
    q = @(x) g(x)/integral(g, LeftBound, RightBound);
    plot(xs, q(xs));
end


