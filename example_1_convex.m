%% Example 1: convex coupled constraint
% Outputs: results/example1_convex beside this script.

clearvars; close all; clc;
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end
out_dir = fullfile(script_dir, 'results', 'example1_convex');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

m = shared_model();
p = proof_parameters();
[y0, init, cert] = initialization_and_certificate(m);
audit = theorem_audit(m, p, cert);

fprintf('EXAMPLE 1: CONVEX COUPLED CONSTRAINT\n');
fprintf('Graph: fixed 4-regular graph with 10 nodes\n');
fprintf('lambda_2(L)=%.12f, ||L||_2=%.12f\n', m.lambda2, m.normL);
fprintf('Initial action ranges: x_1 in [%+.4f,%+.4f], x_2 in [%+.4f,%+.4f]\n', ...
    min(init.x0(:,1)), max(init.x0(:,1)), ...
    min(init.x0(:,2)), max(init.x0(:,2)));
fprintf('Heterogeneous target spans: x_1 %.3f, x_2 %.3f; initial G(x)=%.6e\n', ...
    max(m.xstar(:,1))-min(m.xstar(:,1)), ...
    max(m.xstar(:,2))-min(m.xstar(:,2)),sum(convex_g(init.x0,m)));
fprintf('W_1(0)=%.6e, r_1=%.6e\n', cert.W0, cert.r1);
fprintf('Target ||omega^*||=%.6f, 1^T omega^*=%.3e\n', ...
    norm(m.omegastar),sum(m.omegastar));
fprintf('Target KKT infinity residual = %.3e\n',m.target_kkt_residual_inf);
fprintf('Target feasibility/activity residual = %.3e\n',m.target_feasibility_residual);
fprintf('Target complementarity residual = %.3e\n',m.target_complementarity_residual);
fprintf('Certified l_theta=%.6f <= %.2f, l_varphi=%.6f <= %.2f\n', ...
    cert.ltheta_exact, m.ltheta_used, cert.lvarphi_exact, m.lvarphi_used);
disp(audit(:, {'Condition','Value','Lower','Upper','Margin','Pass'}));
writetable(audit, fullfile(out_dir, 'convex_theorem_audit.csv'));
agent=(1:m.N).';
gstar=convex_g(m.xstar,m);
equilibrium_table=table(agent,m.xstar(:,1),m.xstar(:,2),m.r(:,1),m.r(:,2), ...
    m.dconst,gstar,m.lambdastar,m.omegastar, ...
    'VariableNames',{'Agent','xStar1','xStar2','Reference1','Reference2', ...
    'd_i','g_i_at_target','lambdaStar','omegaStar'});
writetable(equilibrium_table,fullfile(out_dir,'convex_equilibrium_targets.csv'));

equilibrium_audit=table( ...
    ["KKT infinity residual";"aggregate feasibility/activity residual"; ...
     "complementarity residual"], ...
    [m.target_kkt_residual_inf;m.target_feasibility_residual; ...
     m.target_complementarity_residual], ...
    'VariableNames',{'Check','Value'});
writetable(equilibrium_audit,fullfile(out_dir,'convex_equilibrium_audit.csv'));
assert(all(audit.Pass), ...
    'Theorem-inequality check failed.');

%% Simulation and numerical checks
t_eval = unique([0,logspace(-6,-1,650),linspace(0.101,8,1500)]);
opts = odeset('RelTol',2e-9,'AbsTol',2e-11,'MaxStep',0.01);
[t,y] = ode15s(@(~,yy) convex_rhs(yy,m,p), t_eval, y0, opts);
traj = decode_trajectory(t,y,m);
metrics = trajectory_metrics(traj,m);
post = post_audit(traj,metrics,m);
disp(post);
writetable(post, fullfile(out_dir, 'convex_post_simulation_audit.csv'));
assert(all(post.Pass), 'A post-simulation numerical check failed.');

make_topology(m,out_dir);
make_figures(traj,metrics,m,cert,out_dir);
save(fullfile(out_dir,'convex_run_data.mat'), ...
    'm','p','init','cert','audit','equilibrium_table','equilibrium_audit','post','traj','metrics','-v7.3');
fprintf('Completed. Results: %s\n',out_dir);

%% Local functions
function m = shared_model()
    m.N = 10; m.n = 2; m.d = 1;

    % Communication graph
    m.A = [ ...
        0 0 0 1 1 0 1 0 0 1; ...
        0 0 1 0 0 0 1 0 1 1; ...
        0 1 0 0 1 0 1 1 0 0; ...
        1 0 0 0 0 1 1 1 0 0; ...
        1 0 1 0 0 1 0 0 1 0; ...
        0 0 0 1 1 0 0 1 1 0; ...
        1 1 1 1 0 0 0 0 0 0; ...
        0 0 1 1 0 1 0 0 0 1; ...
        0 1 0 0 1 1 0 0 0 1; ...
        1 1 0 0 0 0 0 1 1 0];
    assert(isequal(m.A,m.A.'),'Graph must be undirected.');
    assert(all(sum(m.A,2)==4),'Expected a 4-regular graph.');
    m.L = diag(sum(m.A,2))-m.A;
    ev = sort(real(eig(m.L)));
    m.lambda2 = ev(2); m.normL = ev(end);

    % Game parameters and action bounds
    m.q = 3000;
    m.beta = 0.2;
    m.lambda_star = 30.96;
    m.lambda_bar = 61.92;
    m.xlo = [-1.52;-8];
    m.xhi = [ 1.52; 8];

    % Target equilibrium
    x1star = -0.22 + 0.44*((0:m.N-1).'/9);
    x2star = [0.8;-0.7;0.6;-0.5;0.4;-0.3;0.2;-0.1;-0.4;0.0];
    m.xstar = [x1star,x2star];

    % Choose d_i so g_i(x_i^*) = c_i and sum(c_i) = 0.
    m.local_offsets = 0.11*((1:m.N).'-5.5);
    assert(abs(sum(m.local_offsets))<1e-14, ...
        'Target local constraint values must sum to zero.');
    m.dconst = m.local_offsets ...
        -90*m.xstar(:,1)-3*m.xstar(:,1).^2;

    % Cost references satisfy KKT stationarity at the target.
    gradstar = 90+6*m.xstar(:,1);
    constraint_grad_star = [gradstar,zeros(m.N,1)];

    m.r = m.xstar + ...
        (m.beta*(m.L*m.xstar) ...
        +m.lambda_star*constraint_grad_star)/m.q;

    % Zero-sum auxiliary equilibrium: L*omega^* = c.
    m.omegastar = pinv(m.L)*m.local_offsets;
    m.omegastar = m.omegastar-mean(m.omegastar);
    assert(norm(m.L*m.omegastar-m.local_offsets)<1e-10);
    assert(abs(sum(m.omegastar))<1e-12);
    m.lambdastar = m.lambda_star*ones(m.N,1);

    % Constants for the theorem inequalities
    m.mu_theta = 1; m.mu_varphi = 1;
    m.ltheta_used = 2.5; m.lvarphi_used = 1.1;
    m.ell = m.q+m.beta*m.normL;
    m.tilde_ell = sqrt((m.q+m.beta*4)^2+4*m.beta^2);
    m.mu_x = m.q;

    % |90 + 6*x_{i,1}| <= 99.12 on the action interval.
    m.lg = 100;
    m.l_grad_g = 6;

    tau_exact = min(gradstar.^2);
    m.tau = 7864;
    assert(m.tau<=tau_exact, ...
        'Reported tau is not a valid lower bound at the target.');

    % Slater condition and target residuals
    xhat = m.xstar - repmat([1e-3,0],m.N,1);
    m.slater_margin = -sum(convex_g(xhat,m));
    m.target_interior_margin = min([ ...
        min(m.xstar-m.xlo.',[],'all'), ...
        min(m.xhi.'-m.xstar,[],'all'), ...
        m.lambda_star,m.lambda_bar-m.lambda_star]);

    target_kkt = m.q*(m.xstar-m.r)+m.beta*(m.L*m.xstar)+ ...
        m.lambda_star*constraint_grad_star;
    target_activity = sum(convex_g(m.xstar,m));

    m.target_kkt_residual_inf = norm(target_kkt(:),inf);
    m.target_feasibility_residual = abs(target_activity);
    m.target_complementarity_residual = ...
        abs(m.lambda_star*target_activity);

    m.target_kkt_margin = 1e-10-norm(target_kkt,'fro');
    m.target_activity_margin = 1e-10-abs(target_activity);

    assert(m.target_kkt_residual_inf<1e-10, ...
        'Target KKT stationarity is not exact.');
    assert(m.target_feasibility_residual<1e-10, ...
        'Target shared constraint is not active.');

    m.target_heterogeneity = min_pairwise_distance(m.xstar);
    m.target_omega_norm = norm(m.omegastar);
    m.tau_exact = tau_exact;
end

function p = proof_parameters()
    % Appendix C, Table II
    p.sigma = 0.5;
    p.k = 1.66e5;
    p.rho = 2e-8;
    p.delta1 = p.rho^2;
    p.delta2 = p.k*p.rho^3;
    p.epsilon1 = 1/p.rho;
    p.epsilon2 = 1/p.rho;
    p.epsilon3 = 1/p.rho;
    p.gamma = 5078.131042386909;
end

function [y0,init,cert] = initialization_and_certificate(m)
    N=m.N; n=m.n; k=(1:N).';
    b1=sin(k*sqrt(2))+0.25*cos(k*sqrt(7));
    b1=b1-mean(b1); b1=b1/max(abs(b1));
    b2=cos(k*sqrt(3))-0.2*sin(k*sqrt(5));
    b2=b2-mean(b2); b2=b2/max(abs(b2));
    [ii,jj]=ndgrid(1:N,1:N);
    e1=sin(ii*sqrt(3)+jj*sqrt(5)); e1=e1/max(abs(e1(:)));
    e2=cos(ii*sqrt(5)-jj*sqrt(2)); e2=e2/max(abs(e2(:)));

    % Deterministic, strictly interior initial conditions
    x0=m.xstar+[0.10*b1,0.40*b2];
    assert(all(x0>m.xlo.' & x0<m.xhi.','all'));
    X=zeros(n*N,N);
    for i=1:N
        for j=1:N
            rows=(j-1)*n+(1:n);
            if i==j
                X(rows,i)=x0(j,:).';
            else
                X(rows,i)=m.xstar(j,:).'+[0.003*e1(j,i);0.012*e2(j,i)];
            end
        end
    end
    lambda0=m.lambda_star*(1+0.001*b1);
    omega0=zeros(N,1);

    Q=X;
    for i=1:N
        rows=(i-1)*n+(1:n);
        Q(rows,i)=fermi_inverse(x0(i,:).',m.xlo,m.xhi,m.mu_theta);
    end
    nu0=fermi_inverse(lambda0,0,m.lambda_bar,m.mu_varphi);
    y0=[Q(:);nu0;omega0];

    Waction=0;
    for i=1:N
        Waction=Waction+sum(fermi_D(m.xstar(i,:).',x0(i,:).', ...
            m.xlo,m.xhi,m.mu_theta));
    end
    Woff=0;
    for i=1:N
        for j=1:N
            if i~=j
                rows=(j-1)*n+(1:n);
                Woff=Woff+0.5*sum((X(rows,i)-m.xstar(j,:).').^2);
            end
        end
    end
    Wlambda=sum(fermi_D(m.lambda_star*ones(N,1),lambda0, ...
        zeros(N,1),m.lambda_bar*ones(N,1),m.mu_varphi));
    Womega=0.5*sum((omega0-m.omegastar).^2);
    cert.W0=Waction+Woff+Wlambda+Womega;
    cert.r1=2.04*cert.W0;
    cert.barR=0.5*cert.r1;
    cert.level=cert.r1;

    curv=zeros(N,n);
    for i=1:N
        for ell=1:n
            curv(i,ell)=fermi_level_curvature(m.xstar(i,ell), ...
                m.xlo(ell),m.xhi(ell),m.mu_theta,cert.level);
        end
    end
    cert.ltheta_exact=max(curv,[],'all');
    cert.lvarphi_exact=fermi_level_curvature( ...
        m.lambda_star,0,m.lambda_bar,m.mu_varphi,cert.level);
    cert.initial_boundary_distance=min([ ...
        min(x0-m.xlo.',[],'all'),min(m.xhi.'-x0,[],'all')]);
    assert(cert.ltheta_exact<=m.ltheta_used, ...
        'Pre-simulation l_theta bound is too small.');
    assert(cert.lvarphi_exact<=m.lvarphi_used, ...
        'Pre-simulation l_varphi bound is too small.');
    init=struct('x0',x0,'X0',X,'lambda0',lambda0,'omega0',omega0);
end

function T = theorem_audit(m,p,cert)
    lt=m.ltheta_used; lv=m.lvarphi_used;
    mt=m.mu_theta; mv=m.mu_varphi;

    aomega=m.lg^2*m.normL^2/(2*mv^2);
    kmin=aomega/m.lambda2;
    domega=p.k*m.lambda2-aomega;
    gamma0=(m.tilde_ell+(m.ell+m.tilde_ell)^2/(2*m.mu_x))/m.lambda2;

    Cmin=0.5*min([ ...
        min(mt,1)-m.lg^2*p.rho^3, ...
        mv-p.rho-p.k^2*lv^2*p.rho^5, ...
        1-p.rho]);

    Hgamma=m.ell^2/m.N+m.tilde_ell^2+ ...
        p.gamma*m.normL^2/(2*m.lambda2)+ ...
        m.N*m.d*m.lambda_bar^2*m.l_grad_g^2/2;
    Qgamma=m.lg^2*Hgamma/mt^2+p.k*lv*m.normL;

    A=m.mu_x/m.N-2*p.rho-(m.lg^2/mv)*p.rho^2;
    B=(1-p.rho/2)*p.gamma*m.lambda2-m.tilde_ell-2*p.rho- ...
        (m.lg^2/mv)*p.rho^2;
    Psi=B-(m.ell+m.tilde_ell)^2/(4*m.N*A);
    C=p.rho^2*(m.tau/lt-Qgamma*p.rho);
    D=p.rho^3*(domega-(p.k^2*m.lg^2/2)*p.rho^2);

    cross=-(m.ell+m.tilde_ell)/(2*sqrt(m.N));
    M=blkdiag([A cross;cross B],C,D);
    minM=min(eig(M));

    cond={ ...
        'graph algebraic connectivity'; ...
        'uniform Slater margin (manuscript witness)'; ...
        'target interior margin'; ...
        'target KKT construction margin'; ...
        'active target margin'; ...
        'Bregman sublevel r1 > W1(0)'; ...
        'regional l_theta <= 2.5'; ...
        'regional l_varphi <= 1.1'; ...
        'gamma > gamma0'; ...
        'k > a_omega/lambda2'; ...
        'rho > 0'; ...
        'C_min(rho) > 0'; ...
        'A_rho > 0'; ...
        'Psi_gamma(rho) > 0'; ...
        'C_rho/rho^2 > 0'; ...
        'D_rho/rho^3 > 0'; ...
        'lambda_min(M_rho) > 0'};

    val=[m.lambda2; m.slater_margin; m.target_interior_margin; ...
        m.target_kkt_margin; m.target_activity_margin; ...
        cert.r1; cert.ltheta_exact; cert.lvarphi_exact; ...
        p.gamma; p.k; p.rho; Cmin; A; Psi; ...
        C/p.rho^2; D/p.rho^3; minM];
    lo=[0;0;0;0;0;cert.W0;0;0;gamma0;kmin;0;0;0;0;0;0;0];
    hi=[inf;inf;inf;inf;inf;inf;lt;lv;inf;inf;inf;inf;inf;inf;inf;inf;inf];
    pass=val>lo & val<hi;

    margin=min((val-lo)./max([abs(val),abs(lo),ones(size(val))*1e-14],[],2), ...
        (hi-val)./max([abs(val),abs(hi),ones(size(val))*1e-14],[],2));
    margin(isinf(hi))=(val(isinf(hi))-lo(isinf(hi)))./ ...
        max(abs(val(isinf(hi))),abs(lo(isinf(hi)))+1e-14);

    T=table(string(cond),val,lo,hi,margin,pass, ...
        'VariableNames',{'Condition','Value','Lower','Upper','Margin','Pass'});

    fprintf(['Table-II certificate: gamma0=%.12e, k_min=%.12e, ' ...
        'Cmin=%.12e, A=%.12e, Psi=%.12e, C/rho^2=%.12e, D/rho^3=%.12e\n'], ...
        gamma0,kmin,Cmin,A,Psi,C/p.rho^2,D/p.rho^3);
end

function dy = convex_rhs(y,m,p)
    N=m.N; n=m.n; nq=n*N*N;
    Q=reshape(y(1:nq),n*N,N);
    nu=y(nq+(1:N)); omega=y(nq+N+(1:N));
    [X,x]=reconstruct(Q,m);
    lambda=fermi_map(nu,0,m.lambda_bar,m.mu_varphi);
    dQ=zeros(size(Q));
    for i=1:N
        rowsi=(i-1)*n+(1:n); xi=x(i,:).';
        Fi=m.q*(xi-m.r(i,:).');
        for j=find(m.A(i,:))
            rowsj=(j-1)*n+(1:n);
            Fi=Fi+m.beta*(xi-X(rowsj,i));
        end
        own_cons=zeros(n,1);
        for j=find(m.A(i,:))
            own_cons=own_cons+(xi-X(rowsi,j));
        end
        grad_gi=[90+6*xi(1);0];
        dQ(rowsi,i)=-Fi-grad_gi*lambda(i)-p.gamma*own_cons;
        for player=1:N
            if player==i, continue; end
            rows=(player-1)*n+(1:n);
            v=zeros(n,1);
            for j=find(m.A(i,:))
                v=v+(X(rows,i)-X(rows,j));
            end
            dQ(rows,i)=-p.gamma*v;
        end
    end
    g=convex_g(x,m);
    dnu=g-m.L*omega;
    domega=m.L*lambda;
    dy=[dQ(:);dnu;domega];
end

function [X,x] = reconstruct(Q,m)
    X=Q; x=zeros(m.N,m.n);
    for i=1:m.N
        rows=(i-1)*m.n+(1:m.n);
        xi=fermi_map(Q(rows,i),m.xlo,m.xhi,m.mu_theta);
        X(rows,i)=xi; x(i,:)=xi.';
    end
end

function tr = decode_trajectory(t,y,m)
    nt=numel(t); N=m.N; n=m.n; nq=n*N*N;
    tr.t=t; tr.x=zeros(nt,N,n); tr.lambda=zeros(nt,N);
    tr.omega=y(:,nq+N+(1:N)); tr.esterr=zeros(nt,1);
    tr.esterr_agent=zeros(nt,N);
    tr.stateerr=zeros(nt,1); tr.actionerr=zeros(nt,1);
    tr.lambdaerr=zeros(nt,1); tr.omegaerr=zeros(nt,1); tr.totalerr=zeros(nt,1);
    tr.constraint=zeros(nt,1);
    tr.boundary_distance=zeros(nt,1); tr.primal_hessian_max=zeros(nt,1);
    xstar_stack=reshape(m.xstar.',[],1);
    Xstar=repmat(xstar_stack,1,N);
    for k=1:nt
        Q=reshape(y(k,1:nq),n*N,N);
        [X,x]=reconstruct(Q,m);
        tr.x(k,:,:)=reshape(x,1,N,n);
        la=fermi_map(y(k,nq+(1:N)).',0,m.lambda_bar,m.mu_varphi).';
        tr.lambda(k,:)=la;
        xstack=reshape(x.',[],1);
        Xtruth=repmat(xstack,1,N);
        tr.esterr(k)=norm(X-Xtruth,'fro');
        for agent=1:N
            tr.esterr_agent(k,agent)=norm(X(:,agent)-xstack);
        end
        tr.actionerr(k)=norm(x-m.xstar,'fro');
        tr.lambdaerr(k)=norm(la-m.lambda_star);
        tr.omegaerr(k)=norm(tr.omega(k,:).'-m.omegastar);
        tr.totalerr(k)=tr.actionerr(k)+tr.esterr(k)+tr.lambdaerr(k)+tr.omegaerr(k);
        tr.stateerr(k)=tr.totalerr(k);
        tr.constraint(k)=sum(convex_g(x,m));
        tr.boundary_distance(k)=min([ ...
            min(x-m.xlo.',[],'all'),min(m.xhi.'-x,[],'all')]);
        tr.primal_hessian_max(k)=max_fermi_hessian(x,m.xlo,m.xhi,m.mu_theta);
    end
end

function met = trajectory_metrics(tr,m)
    met.normalized_error=tr.totalerr/max(tr.totalerr(1),realmin);
    met.log_error=log(max(met.normalized_error,1e-14));
    met.log_estimate=log(max(tr.esterr/max(tr.esterr(1),realmin),1e-14));
    met.consensus=max(abs(tr.lambda-mean(tr.lambda,2)),[],2);
    met.log_consensus=log(max(met.consensus/max(met.consensus(1),realmin),1e-14));
    met.omega_error=sqrt(sum((tr.omega-m.omegastar.').^2,2));
    met.log_omega_error=log(max(met.omega_error/max(met.omega_error(1),realmin),1e-14));
    idx=tr.t>=0.35 & met.normalized_error>1e-10;
    if nnz(idx)>20
        c=polyfit(tr.t(idx),met.log_error(idx),1);
        met.fit_rate=-c(1); met.log_fit=polyval(c,tr.t);
    else
        met.fit_rate=NaN; met.log_fit=nan(size(tr.t));
    end
    xend=squeeze(tr.x(end,:,:));
    lambda_end=tr.lambda(end,:).';
    grad_end=90+6*xend(:,1);
    constraint_term=[lambda_end.*grad_end,zeros(m.N,1)];
    met.kkt_terminal=norm(m.q*(xend-m.r)+m.beta*(m.L*xend)+ ...
        constraint_term,'fro');
end

function T = post_audit(tr,met,m)
    name=["terminal normalized state error"; ...
        "terminal estimate error";"terminal multiplier disagreement"; ...
        "terminal active-constraint residual";"terminal KKT residual"; ...
        "zero-sum omega residual";"terminal omega-target residual"; ...
        "positive fitted exponential rate"; ...
        "minimum action-boundary distance"; ...
        "trajectory Fermi Hessian below certified bound"; ...
        "heterogeneous equilibrium separation"; ...
        "nonzero zero-sum auxiliary equilibrium"];
    value=[met.normalized_error(end);tr.esterr(end);met.consensus(end); ...
        abs(tr.constraint(end));met.kkt_terminal; ...
        max(abs(sum(tr.omega,2)));met.omega_error(end);met.fit_rate; ...
        min(tr.boundary_distance);max(tr.primal_hessian_max); ...
        m.target_heterogeneity;m.target_omega_norm];
    tol=[3e-5;2e-5;2e-5;2e-5;2e-2;2e-8;2e-5;0; ...
        1e-3;m.ltheta_used;1e-3;1e-3];
    pass=[value(1:7)<=tol(1:7);value(8)>tol(8);value(9)>tol(9); ...
        value(10)<=tol(10);value(11)>tol(11);value(12)>tol(12)];
    T=table(name,value,tol,pass,'VariableNames',{'Check','Value','Tolerance','Pass'});
end

function make_topology(m,out_dir)
    f=figure('Color','w','Position',[100 100 620 520]);
    G=graph(m.A);

    p=plot(G,'Layout','circle', ...
        'LineWidth',1.35, ...
        'EdgeColor',[76,92,104]/255, ...
        'NodeColor','w', ...
        'MarkerSize',1, ...
        'NodeLabel',repmat({''},1,m.N));

    hold on;

    scatter(p.XData,p.YData,300, ...
        'MarkerFaceColor','w', ...
        'MarkerEdgeColor',[47,72,88]/255, ...
        'LineWidth',1.6);

    for i=1:m.N
        text(p.XData(i),p.YData(i),sprintf('%d',i), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'Color','k', ...
            'FontName','Times New Roman', ...
            'FontSize',11, ...
            'FontWeight','normal');
    end

    axis equal;
    axis off;
    save_figure_bundle(f,out_dir,'shared_topology');
    close(f);
end

function make_figures(tr,met,m,cert,out_dir) %#ok<INUSD>
    cols=agent_colors();
    blue=cols(1,:);
    red=cols(4,:);
    nt=numel(tr.t);

    aggregate_action_error=zeros(nt,1);
    for i=1:m.N
        dx=squeeze(tr.x(:,i,:))-m.xstar(i,:);
        aggregate_action_error=aggregate_action_error+sum(dx.^2,2);
    end
    aggregate_estimate_error=sum(tr.esterr_agent.^2,2);

    % Primal and decision-estimate errors
    f=figure('Color','w','Position',[100 100 1260 480]);
    tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');

    ax=nexttile(tl); hold(ax,'on');
    semilogy(ax,tr.t,max(aggregate_action_error,1e-16), ...
        'Color',blue,'LineWidth',2.0, ...
        'DisplayName','$\sum_{i=1}^{N}\|x_i-x_i^\circ\|^2$');
    grid(ax,'on');
    xlabel(ax,'$t$','Interpreter','latex');
    ylabel(ax,'$\sum_{i=1}^{N}\|x_i(t)-x_i^\circ\|^2$', ...
        'Interpreter','latex');
    legend(ax,'Location','best','Interpreter','latex');

    ax=nexttile(tl); hold(ax,'on');
    semilogy(ax,tr.t,max(aggregate_estimate_error,1e-16), ...
        'Color',red,'LineWidth',2.0, ...
        'DisplayName','$\sum_{i=1}^{N}\|\mathbf{x}^i-\mathrm{col}\{x_j\}\|^2$');
    grid(ax,'on');
    xlabel(ax,'$t$','Interpreter','latex');
    ylabel(ax,['$\sum_{i=1}^{N}\|\mathbf{x}^i(t)-' ...
        '\mathrm{col}\{x_j(t)\}_{j=1}^{N}\|^2$'], ...
        'Interpreter','latex');
    legend(ax,'Location','best','Interpreter','latex');

    format_axes(f);
    save_figure_bundle(f,out_dir,'convex_tra');
    close(f);

    % Multipliers and auxiliary states
    f=figure('Color','w','Position',[80 100 1460 500]);
    tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');

    ax=nexttile(tl); hold(ax,'on');
    hlam=gobjects(m.N,1);
    for i=1:m.N
        hlam(i)=plot(ax,tr.t,tr.lambda(:,i),'Color',cols(i,:), ...
            'LineWidth',1.40,'DisplayName',sprintf('agent %d',i));
    end
    href=yline(ax,m.lambda_star,'k--','LineWidth',1.35, ...
        'DisplayName','$\lambda^\circ$');
    grid(ax,'on');
    xlabel(ax,'$t$','Interpreter','latex');
    ylabel(ax,'$\lambda_i$','Interpreter','latex');
    legend(ax,[hlam;href],'Location','eastoutside','Interpreter','latex');

    ax=nexttile(tl); hold(ax,'on');
    homega=gobjects(m.N,1);
    for i=1:m.N
        homega(i)=plot(ax,tr.t,tr.omega(:,i),'Color',cols(i,:), ...
            'LineWidth',1.40,'DisplayName',sprintf('agent %d',i));
    end
    hsum=plot(ax,tr.t,sum(tr.omega,2),'k--','LineWidth',1.65, ...
        'DisplayName','$\sum_i\omega_i(t)$');
    grid(ax,'on');
    xlabel(ax,'$t$','Interpreter','latex');
    ylabel(ax,'$\omega_i$','Interpreter','latex');
    legend(ax,[homega;hsum],'Location','eastoutside','Interpreter','latex');

    format_axes(f);
    save_figure_bundle(f,out_dir,'convex_aux');
    close(f);

    % Fitted convergence rate and shared constraint
    f=figure('Color','w','Position',[100 100 1260 480]);
    tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');

    ax=nexttile(tl); hold(ax,'on');
    h1=plot(ax,tr.t,met.log_error,'LineWidth',2.0,'Color',blue, ...
        'DisplayName','simulation');
    h2=plot(ax,tr.t,met.log_fit,'--','LineWidth',1.55,'Color',red, ...
        'DisplayName','affine fit');
    grid(ax,'on');
    xlabel(ax,'$t$','Interpreter','latex');
    ylabel(ax,'$\ln(e_{\Sigma,1}(t)/e_{\Sigma,1}(0))$','Interpreter','latex');
    legend(ax,[h1 h2],'Location','southwest','Interpreter','latex');

    ax=nexttile(tl); hold(ax,'on');
    h1=plot(ax,tr.t,tr.constraint,'LineWidth',2.0,'Color',blue, ...
        'DisplayName','$G(x(t))$');
    h2=yline(ax,0,'--','LineWidth',1.35,'Color',red, ...
        'DisplayName','active boundary');
    grid(ax,'on');
    xlabel(ax,'$t$','Interpreter','latex');
    ylabel(ax,'$G(x(t))$','Interpreter','latex');
    legend(ax,[h1 h2],'Location','best','Interpreter','latex');

    format_axes(f);
    save_figure_bundle(f,out_dir,'convex_exp');
    close(f);
end

function dmin = min_pairwise_distance(X)
    dmin=inf;
    for i=1:size(X,1)-1
        for j=i+1:size(X,1)
            dmin=min(dmin,norm(X(i,:)-X(j,:)));
        end
    end
end

function g = convex_g(x,m)
    g=90*x(:,1)+3*x(:,1).^2+m.dconst;
end

function J = convex_grad(x)
    J=[90+6*x(:,1),zeros(size(x,1),1)];
end

function Hmax = max_fermi_hessian(x,lo,hi,mu)
    lo=lo(:).'; hi=hi(:).'; mu=expand(mu,size(x,2)).';
    width=hi-lo; kk=mu.*width/4;
    H=kk.*width./((x-lo).*(hi-x));
    Hmax=max(H,[],'all');
end

function cols=agent_colors()
    cols=[ ...
        31,119,180; ...
        255,127,14; ...
        44,160,44; ...
        214,39,40; ...
        148,103,189; ...
        140,86,75; ...
        227,119,194; ...
        23,190,207; ...
        188,189,34; ...
        127,127,127]/255;
end

function format_axes(f)
    ax=findall(f,'Type','axes');
    set(ax,'FontName','Times New Roman','FontSize',12,'LineWidth',0.9, ...
        'TickLabelInterpreter','latex','Box','on');
end

function save_figure_bundle(f,out_dir,stem)
    savefig(f,fullfile(out_dir,[stem '.fig']));
    exportgraphics(f,fullfile(out_dir,[stem '.png']),'Resolution',320);
    try
        exportgraphics(f,fullfile(out_dir,[stem '.eps']),'ContentType','vector');
    catch
        print(f,fullfile(out_dir,[stem '.eps']),'-depsc','-painters');
    end
end

function x = fermi_map(z,lo,hi,mu)
    z=z(:); lo=expand(lo,numel(z)); hi=expand(hi,numel(z));
    width=hi-lo; kk=mu.*width/4; s=zeros(size(z));
    a=z./kk; pos=a>=0;
    s(pos)=1./(1+exp(-min(a(pos),700)));
    ea=exp(max(a(~pos),-700)); s(~pos)=ea./(1+ea);
    x=lo+width.*s;
end

function z = fermi_inverse(x,lo,hi,mu)
    x=x(:); lo=expand(lo,numel(x)); hi=expand(hi,numel(x));
    width=hi-lo; kk=mu.*width/4;
    p=min(max((x-lo)./width,1e-14),1-1e-14);
    z=kk.*log(p./(1-p));
end

function D = fermi_D(ref,current,lo,hi,mu)
    ref=ref(:); current=current(:); lo=expand(lo,numel(ref));
    hi=expand(hi,numel(ref)); width=hi-lo; kk=mu.*width/4;
    pr=(ref-lo)./width; pc=(current-lo)./width;
    pr=min(max(pr,1e-14),1-1e-14);
    pc=min(max(pc,1e-14),1-1e-14);
    D=kk.*width.*(pr.*log(pr./pc)+(1-pr).*log((1-pr)./(1-pc)));
end

function c = fermi_level_curvature(ref,lo,hi,mu,level)
    width=hi-lo; kk=mu*width/4;
    fun=@(x) fermi_D(ref,x,lo,hi,mu)-level;
    tiny=1e-13*width;
    left=fzero(fun,[lo+tiny,ref-1e-15*width]);
    right=fzero(fun,[ref+1e-15*width,hi-tiny]);
    H=@(x) kk*width/((x-lo)*(hi-x));
    c=max(H(left),H(right));
end

function v = expand(v,n)
    if isscalar(v), v=repmat(v,n,1); else, v=v(:); end
end
