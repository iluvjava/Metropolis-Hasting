f = @(x) sin(pi*x)./(pi*x);                            % the objective function. 
LeftBound = -3; 
RightBound = 3;
N = 10000;                                    % Number of sample points. 
xs = linspace(LeftBound, RightBound, N);      % the log space. 
Ts = logspace(1, -1, 5);                      % temperature. 
hold on; 
for T = Ts
    g = @(x) exp(f(x)/T);
    q = @(x) g(x)/integral(g, LeftBound, RightBound);
    plot(xs, q(xs));
end

title("Simulated Annealing temps pproaches zero. ")
legend(arrayfun(@num2str, Ts, 'UniformOutput', 0), 'Location', 'NorthWest');
saveas(gcf, 'sa_temp.png')