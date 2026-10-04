%% Example 2: composite nonconvex coupled constraint
% Outputs: results/example2_nonconvex beside this script.

clearvars; close all; clc;
script_dir=fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir=pwd; end
out_dir=fullfile(script_dir,'results','example2_nonconvex');
if ~exist(out_dir,'dir'), mkdir(out_dir); end

t_final=30;

m=shared_model();
p=proof_parameters();
[y0,init,cert]=initialization_and_certificate(m);
audit=theorem_audit(m,p,cert);

fprintf('EXAMPLE 2: COMPOSITE NONCONVEX CONSTRAINT\n');
fprintf('Graph: fixed 4-regular graph with 10 nodes\n');
fprintf('lambda_2(L)=%.12f, ||L||_2=%.12f\n',m.lambda2,m.normL);
fprintf('Target multiplier lambda^*=%.4f\n',m.lambda_star);
fprintf('Integration horizon T=%.1f.\n',t_final);
fprintf('Initial action ranges: x_1 [%+.4f,%+.4f], x_2 [%+.4f,%+.4f]\n', ...
    min(init.x0(:,1)),max(init.x0(:,1)),min(init.x0(:,2)),max(init.x0(:,2)));
fprintf('Heterogeneous target spans: x_1 %.3f, x_2 %.3f; initial original G(x)=%.6e\n', ...
    max(m.xstar(:,1))-min(m.xstar(:,1)), ...
    max(m.xstar(:,2))-min(m.xstar(:,2)),aggregate_original_g(init.x0,m));
fprintf('Target ||omega^*||=%.6f, 1^T omega^*=%.3e\n', ...
    norm(m.omegastar),sum(m.omegastar));
fprintf('W_2(0)=%.12e, r_2=%.12e, 2r_2=%.12e\n', ...
    cert.W0,cert.r2,cert.level);
fprintf(['Certified on K_{2,2r_2}: l_theta=%.9f <= %.2f, ' ...
    'l_varphi=%.9f <= %.2f, l_phi=%.9f <= %.2f\n'], ...
    cert.ltheta_exact,m.ltheta_used,cert.lvarphi_exact,m.lvarphi_used, ...
    cert.lphi_exact,m.lphi_used);
fprintf('Whole-domain L_tilde_g_x=%.9g, L_tilde_g_eta=%.9g, L_G_x=%.9g\n', ...
    m.Lx,m.Leta,m.LGx);
fprintf('Original-constraint Hessian test: minimum eigenvalue %.6e<0\n', ...
    m.nonconvex_hessian_min);
disp(audit(:,{'Condition','Value','Lower','Upper','Margin','Pass'}));
writetable(audit,fullfile(out_dir,'nonconvex_theorem_audit.csv'));
constant_name=["L_tilde_g_x";"L_tilde_g_eta";"L_G_x";"bar_rho_x"; ...
    "mu_G";"l_Phi";"l_G";"iota";"negative_Hessian_eigenvalue"; ...
    "W2_initial";"r2";"two_r2"; ...
    "certified_l_theta_K2_2r";"certified_l_varphi_K2_2r"; ...
    "certified_l_phi_K2_2r"];
constant_value=[m.Lx;m.Leta;m.LGx;m.bar_rho_x;m.mu_G;m.lPhi;m.lG; ...
    m.iota;m.nonconvex_hessian_min;cert.W0;cert.r2;cert.level; ...
    cert.ltheta_exact;cert.lvarphi_exact;cert.lphi_exact];
model_constants=table(constant_name,constant_value, ...
    'VariableNames',{'Constant','Value'});
writetable(model_constants,fullfile(out_dir,'nonconvex_model_constants.csv'));
agent=(1:m.N).';
equilibrium_table=table(agent,m.xstar(:,1),m.xstar(:,2),m.r(:,1),m.r(:,2), ...
    m.local_offsets,m.lambdastar,m.omegastar, ...
    'VariableNames',{'Agent','xStar1','xStar2','Reference1','Reference2', ...
    'LocalConstraintAtTarget','lambdaStar','omegaStar'});
writetable(equilibrium_table,fullfile(out_dir,'nonconvex_equilibrium_targets.csv'));
assert(all(audit.Pass), ...
    'Theorem-inequality check failed.');

%% Canonical mirror dynamics
t_eval=unique([0,logspace(-9,-2,700),linspace(0.0101,t_final,2300)]);

opts=odeset('RelTol',2e-9,'AbsTol',2e-11,'MaxStep',0.02);
[tp,yp]=ode15s(@(~,yy) proposed_rhs(yy,m,p),t_eval,y0,opts);
proposed=decode_proposed(tp,yp,m);
pm=proposed_metrics(proposed,m);

% Solver sensitivity: smaller tolerances and maximum step size.
opts_tight=odeset('RelTol',2e-10,'AbsTol',2e-12,'MaxStep',0.01);
[tp_tight,yp_tight]=ode15s(@(~,yy) proposed_rhs(yy,m,p),t_eval,y0,opts_tight);
proposed_tight=decode_proposed(tp_tight,yp_tight,m);
pm_tight=proposed_metrics(proposed_tight,m);

%% Two-player annular diagnostic
% This separate example is not part of the ten-agent theorem certificate.
diagnostic=run_annular_globality_diagnostic();
writetable(diagnostic.summary,fullfile(out_dir,'annular_globality_summary.csv'));
write_globality_report(diagnostic,out_dir);

post=post_audit(proposed,pm,m);
disp(post);
writetable(post,fullfile(out_dir,'nonconvex_post_simulation_audit.csv'));
assert(all(post.Pass),'A post-simulation numerical check failed.');

%% Compare baseline and tightened solver settings
post_tight=post_audit(proposed_tight,pm_tight,m);
solver_sensitivity=table(post.Check,post.Value,post_tight.Value, ...
    abs(post_tight.Value-post.Value),post.Tolerance,post.Pass,post_tight.Pass, ...
    'VariableNames',{'Check','Baseline','Tightened','AbsDifference', ...
    'Tolerance','BaselinePass','TightenedPass'});
disp(solver_sensitivity);
writetable(solver_sensitivity, ...
    fullfile(out_dir,'nonconvex_solver_sensitivity.csv'));

x_end=squeeze(proposed.x(end,:,:));
x_end_tight=squeeze(proposed_tight.x(end,:,:));
eta_end=squeeze(proposed.eta(end,:,:));
eta_end_tight=squeeze(proposed_tight.eta(end,:,:));
terminal_quantity=["x terminal Frobenius difference"; ...
    "lambda terminal 2-norm difference"; ...
    "omega terminal 2-norm difference"; ...
    "eta terminal Frobenius difference"; ...
    "original-constraint terminal difference"; ...
    "canonical-constraint terminal difference"; ...
    "consistency-residual terminal difference"; ...
    "KKT-residual terminal difference"];
terminal_difference=[norm(x_end_tight-x_end,'fro'); ...
    norm(proposed_tight.lambda(end,:)-proposed.lambda(end,:)); ...
    norm(proposed_tight.omega(end,:)-proposed.omega(end,:)); ...
    norm(eta_end_tight-eta_end,'fro'); ...
    abs(proposed_tight.original_constraint(end)-proposed.original_constraint(end)); ...
    abs(proposed_tight.canonical_constraint(end)-proposed.canonical_constraint(end)); ...
    abs(proposed_tight.canonical_consistency(end)-proposed.canonical_consistency(end)); ...
    abs(pm_tight.kkt_terminal-pm.kkt_terminal)];
terminal_sensitivity=table(terminal_quantity,terminal_difference, ...
    'VariableNames',{'Quantity','AbsDifference'});
disp(terminal_sensitivity);
writetable(terminal_sensitivity, ...
    fullfile(out_dir,'nonconvex_terminal_sensitivity.csv'));

make_figures(proposed,pm,m,cert,out_dir);
make_globality_diagnostic_figures(diagnostic,out_dir);
save(fullfile(out_dir,'nonconvex_run_data.mat'), ...
    'm','p','init','cert','audit','model_constants','equilibrium_table','proposed','pm', ...
    'post_tight','solver_sensitivity','terminal_sensitivity', ...
    'diagnostic','post','-v7.3');
fprintf('Completed. Results: %s\n',out_dir);

%% Local functions
function m=shared_model()
    m.N=10; m.n=2; m.d=1;
    m.A=[ ...
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
    assert(isequal(m.A,m.A.') && all(sum(m.A,2)==4));
    m.L=diag(sum(m.A,2))-m.A;
    ev=sort(real(eig(m.L))); m.lambda2=ev(2); m.normL=ev(end);

    % Game parameters and action bounds
    m.q=2.0e5; m.beta=0.2; m.h=1.0e3;
    m.lambda_star=25.0; m.lambda_bar=2*m.lambda_star;
    assert(abs(m.lambda_star-30.96)>1, ...
        'The nonconvex multiplier must be distinct from the convex example.');
    m.xlo=[-1.52;-8.0]; m.xhi=[1.52;8.0];
    theta=2*pi*(0:m.N-1)'/m.N;
    % Target equilibrium
    x1_target=linspace(-0.65,0.65,m.N).';
    x2_target=1.35*cos(theta)-0.35*sin(2*theta);
    m.xstar=[x1_target, x2_target];
    assert(min(diff(sort(m.xstar(:,1))))>0.14, ...
        'The first-component equilibrium values are not sufficiently separated.');
    m.local_offsets=0.001*(-4.5:1:4.5).';
    assert(abs(sum(m.local_offsets))<1e-14);
    m.omegastar=pinv(m.L)*m.local_offsets;
    m.omegastar=m.omegastar-mean(m.omegastar);
    m.Romega=[eye(m.N-1);-ones(1,m.N-1)];
    m.Romega_pinv=(m.Romega.'*m.Romega)\m.Romega.';
    assert(norm(m.L*m.omegastar-m.local_offsets)<1e-10);
    assert(abs(sum(m.omegastar))<1e-12);
    m.lambdastar=m.lambda_star*ones(m.N,1);

    % For u_i = x_i - x_i^*, rho_i = [b*u_1 + (m/2)||u||^2; A*sin(k*u_2)].
    % Phi_i(rho) = E*rho_1 + (a/2)||rho||^2 + c_i.
    m.E=7.04e6;
    m.a_outer=68;
    m.mu_G=0.2;
    m.wave=100;
    m.b=m.h/m.E;
    m.rho2_derivative_amp=0.12;
    m.rho2_amp=m.rho2_derivative_amp/m.wave;
    m.rho2_curvature=m.rho2_amp*m.wave^2;
    m.eta_half=[0.01;0.01];
    % Ensures Delta_i lies in the interior of E_i^+.
    m.m_inner=4.5455e-8;
    m.etastar=repmat([m.E,0],m.N,1);
    m.etalo=[m.E-m.eta_half(1);-m.eta_half(2)];
    m.etahi=[m.E+m.eta_half(1); m.eta_half(2)];

    m.mu_theta=1; m.mu_varphi=1;
    m.mu_eta=[1;1];
    m.mu_canonical=min(m.mu_eta);
    m.ltheta_used=4.5;
    m.lvarphi_used=1.8;
    m.lphi_used=1.8;

    m.ell=m.q+m.beta*m.normL;
    m.tilde_ell=sqrt((m.q+m.beta*4)^2+4*m.beta^2);
    m.mu_x=m.q;
    m.r=m.xstar+(m.beta*(m.L*m.xstar)+ ...
        m.lambda_star*repmat([m.h,0],m.N,1))/m.q;

    % Whole-domain constants for the theorem inequalities
    umax=zeros(m.N,m.n);
    for i=1:m.N
        umax(i,:)=max(abs(m.xlo.'-m.xstar(i,:)), ...
            abs(m.xhi.'-m.xstar(i,:)));
    end
    m.umax=max(umax,[],1);
    rho1abs=m.b*m.umax(1)+0.5*m.m_inner*sum(m.umax.^2);
    rho_local=hypot(rho1abs,m.rho2_amp);
    m.bar_rho=sqrt(m.N)*rho_local*(1+1e-8);
    m.bar_Phi_star=sqrt(m.N)*hypot( ...
        m.eta_half(1)/m.a_outer,m.eta_half(2)/m.a_outer);
    m.Leta=m.bar_rho+m.bar_Phi_star;
    m.bar_eta=sqrt(m.N)*hypot(m.E+m.eta_half(1),m.eta_half(2));
    base_grad=hypot(m.b+m.m_inner*m.umax(1), ...
        m.m_inner*m.umax(2));
    m.bar_rho_x=sqrt(base_grad^2+m.rho2_derivative_amp^2)*(1+1e-8);
    m.Lx=(m.E+m.eta_half(1))*base_grad+ ...
        m.eta_half(2)*m.rho2_derivative_amp;
    m.LGx=(m.E+m.eta_half(1))*m.m_inner+ ...
        m.eta_half(2)*m.rho2_curvature;
    assert(m.Lx<=1000.71 && m.Leta<=4.578e-3 && ...
        m.LGx<=0.44001 && m.bar_rho_x<=0.120001, ...
        'Reported whole-domain constants are not valid upper bounds.');
    m.lG=m.h;
    m.iota=m.h^2;
    m.lPhi=m.a_outer;

  % Uniform canonical Slater witness used in the manuscript:
% xhat_i = x_i^* - col{1e-3,0}.
worst=zeros(m.N,1);
for i=1:m.N
    xhat=m.xstar(i,:).'-[1e-3;0];

    % The witness must remain in the local feasible action set.
    assert(all(xhat>m.xlo & xhat<m.xhi));

    [rho,~]=rho_map(xhat,i,m);

    % Maximize the canonical constraint over eta_i in Delta_i.
    % Since rho_2(xhat_i)=0, the worst eta_2 is 0 and
    % eta_1 is the clipped unconstrained maximizer E+a*rho_1.
    eta1=max(m.etalo(1), ...
        min(m.etahi(1),m.E+m.a_outer*rho(1)));

    worst(i)=eta1*rho(1) ...
        -(eta1-m.E)^2/(2*m.a_outer) ...
        +m.local_offsets(i);
end

m.slater_margin=-sum(worst);
assert(m.slater_margin>9, ...
    'Manuscript Slater witness does not provide the claimed margin.');

    m.target_interior_margin=min([ ...
        min(m.xstar-m.xlo.',[],'all'), ...
        min(m.xhi.'-m.xstar,[],'all'), ...
        m.lambda_star,m.lambda_bar-m.lambda_star, ...
        m.eta_half(1),m.eta_half(2)]);
    target_kkt=m.q*(m.xstar-m.r)+m.beta*(m.L*m.xstar)+ ...
        m.lambda_star*repmat([m.h,0],m.N,1);
    m.target_consistency_margin=1e-10;
    m.target_kkt_margin=1e-10-norm(target_kkt,'fro');
    m.target_original_activity_margin=1e-10-abs(sum(m.local_offsets));
    m.target_canonical_activity_margin=m.target_original_activity_margin;
    m.target_heterogeneity=min_pairwise_distance(m.xstar);
    m.target_omega_norm=norm(m.omegastar);

    hmin=inf;
    for i=1:m.N
        xnc=m.xstar(i,:).'+[0;pi/(2*m.wave)];
        assert(all(xnc>m.xlo & xnc<m.xhi));
        hmin=min(hmin,min(eig(original_constraint_hessian(xnc,i,m))));
    end
    m.nonconvex_hessian_min=hmin;
    assert(m.nonconvex_hessian_min<-0.5);
end

function p=proof_parameters()
    % Appendix G, Table III
    p.epsilon5=2.75e-4;
    p.epsilon4=5.6e-4;
    p.sigma=1.1e-4;
    p.epsilon6=8.2;
    p.komega=1.84e4;
    p.delta3=2.25e-9;
    p.delta4=p.komega*p.delta3^2;
    p.gamma=3.429329469016026e5;
end

function [y0,init,cert]=initialization_and_certificate(m)
    N=m.N; n=m.n; k=(1:N).';
    b1=sin(k*sqrt(2))+0.25*cos(k*sqrt(7));
    b1=b1-mean(b1); b1=b1/max(abs(b1));
    b2=cos(k*sqrt(3))-0.2*sin(k*sqrt(5));
    b2=b2-mean(b2); b2=b2/max(abs(b2));
    [ii,jj]=ndgrid(1:N,1:N);
    e1=sin(ii*sqrt(3)+jj*sqrt(5)); e1=e1/max(abs(e1(:)));
    e2=cos(ii*sqrt(5)-jj*sqrt(2)); e2=e2/max(abs(e2(:)));

    x0=m.xstar+[6e-4*b1,2e-3*b2];
    assert(all(x0>m.xlo.' & x0<m.xhi.','all'));
    X=zeros(n*N,N);
    for i=1:N
        for j=1:N
            rows=(j-1)*n+(1:n);
            if i==j
                X(rows,i)=x0(j,:).';
            else
                X(rows,i)=m.xstar(j,:).'+[3e-5*e1(j,i);1.2e-4*e2(j,i)];
            end
        end
    end
    lambda0=m.lambda_star*(1+1e-5*b1);
    omega0=zeros(N,1);
    zeta0=omega0(1:N-1);
    eta0=[m.E+0.006*m.eta_half(1)*b1, ...
        0.006*m.eta_half(2)*b2];
    assert(all(eta0(:,1)>m.etalo(1) & eta0(:,1)<m.etahi(1)));
    assert(all(eta0(:,2)>m.etalo(2) & eta0(:,2)<m.etahi(2)));

    Q=X;
    for i=1:N
        rows=(i-1)*n+(1:n);
        Q(rows,i)=fermi_inverse(x0(i,:).',m.xlo,m.xhi,m.mu_theta);
    end
    nu0=fermi_inverse(lambda0,0,m.lambda_bar,m.mu_varphi);
    alpha0=zeros(2,N);
    for i=1:N
        alpha0(:,i)=fermi_inverse(eta0(i,:).',m.etalo,m.etahi,m.mu_eta);
    end
    y0=[Q(:);nu0;zeta0;alpha0(:)];

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
    Weta=0;
    for i=1:N
        Weta=Weta+sum(fermi_D(m.etastar(i,:).',eta0(i,:).', ...
            m.etalo,m.etahi,m.mu_eta));
    end
    cert.W0=Waction+Woff+Wlambda+Womega+Weta;
    cert.r2=1.02*cert.W0;
    cert.barR=cert.r2;
    cert.level=2*cert.r2;
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
    cert.lphi1_exact=fermi_level_curvature( ...
        m.E,m.etalo(1),m.etahi(1),m.mu_eta(1),cert.level);
    cert.lphi2_exact=fermi_level_curvature( ...
        0,m.etalo(2),m.etahi(2),m.mu_eta(2),cert.level);
    cert.lphi_exact=max(cert.lphi1_exact,cert.lphi2_exact);
    cert.initial_boundary_distance=min([ ...
        min(x0-m.xlo.',[],'all'),min(m.xhi.'-x0,[],'all')]);
    assert(cert.ltheta_exact<=m.ltheta_used, ...
        'Primal mirror certificate failed: l_theta=%.9g exceeds %.9g.', ...
        cert.ltheta_exact,m.ltheta_used);
    assert(cert.lvarphi_exact<=m.lvarphi_used, ...
        'Multiplier mirror certificate failed: l_varphi=%.9g exceeds %.9g.', ...
        cert.lvarphi_exact,m.lvarphi_used);
    assert(cert.lphi_exact<=m.lphi_used, ...
        'Canonical mirror certificate failed: l_phi=%.9g exceeds %.9g.', ...
        cert.lphi_exact,m.lphi_used);
    init=struct('x0',x0,'X0',X,'lambda0',lambda0, ...
        'omega0',omega0,'zeta0',zeta0,'eta0',eta0);
end

function T=theorem_audit(m,p,cert)
    lt=m.ltheta_used; lv=m.lvarphi_used;
    mt=m.mu_theta; mv=m.mu_varphi;
    mu2=min([m.mu_theta,m.mu_varphi,m.mu_canonical,1]);

    lambdaSigma=m.lambda_star;
    lambdaMin=m.lambda_star;
    mu_c=m.mu_x+m.mu_G*lambdaSigma- ...
        m.N*m.LGx*lambdaSigma/2;
    d_eta=lambdaMin/m.lPhi;
    c_eta=m.N*m.d*m.lambda_bar^2*m.bar_rho_x^2;
    core_lhs=mu_c*d_eta;
    core_rhs=m.N*c_eta;

    % Sequential certificate: Steps 1--3
    eps5_lo=m.N/mu_c;
    eps5_hi=d_eta/c_eta;
    mu5=mu_c-m.N/p.epsilon5;
    e0=d_eta-p.epsilon5*c_eta;
    eps4_min=5*m.N/(2*mu5);
    mu4=mu5-5*m.N/(2*p.epsilon4);
    sigma_max=min([1,p.epsilon4,mu4/(m.mu_G*lambdaSigma)]);
    a0=mu4-p.sigma*m.mu_G*lambdaSigma;
    eps6_min=2/e0;
    e1=e0-2/p.epsilon6;

    bomega=p.epsilon4*m.lG^2*m.normL^2/(2*mv^2);
    komega_min=bomega/m.lambda2;
    domega=p.komega*m.lambda2-bomega;
    qomega=p.epsilon4*m.Lx^2+p.epsilon6*m.Leta^2;

    % Step 4 and invariance factor
    d3_cond1=a0/m.N-p.delta3*m.lG*m.Lx/mv;
    d3_cond2=e1-p.epsilon4*p.delta3^2*m.lG^2*m.Leta^2/(2*mv^2);
    d3_cond3=domega-(p.komega^2*qomega/2)*p.delta3^2;
    varrho2=(p.delta3*m.lG+p.komega*p.delta3^2*lv)/mu2;
    delta4_error=abs(p.delta4-p.komega*p.delta3^2);

    % Steps 5--6
    beta_x=m.LGx*lambdaSigma/2+ ...
        p.delta3*m.lG*m.Lx/mv+5/(2*p.epsilon4)+1/p.epsilon5;
    Atilde=m.mu_x/m.N+(1-p.sigma)*m.mu_G*lambdaSigma/m.N-beta_x;

    gammabar=(m.tilde_ell+(1/p.sigma-1)*m.mu_G*lambdaSigma+ ...
        beta_x+(m.ell+m.tilde_ell)^2/(4*m.N*Atilde))/ ...
        ((1-p.sigma/p.epsilon4)*m.lambda2);

    QF=@(gam) (m.lG^2/mt^2)*( ...
        p.epsilon4*(m.ell^2/m.N+m.tilde_ell^2+ ...
        gam*m.normL^2/(4*p.sigma*m.lambda2)+ ...
        m.N*m.d*m.lambda_bar^2*m.LGx^2/2) + ...
        p.epsilon6*c_eta/2 );
    Qlambda=@(gam) QF(gam)+p.komega*lv*m.normL;
    iotareq=lt*(p.epsilon6*m.Leta^2/p.delta3+ ...
        p.delta3*Qlambda(gammabar));

    Btilde=(1-p.sigma/p.epsilon4)*p.gamma*m.lambda2-m.tilde_ell- ...
        (1/p.sigma-1)*m.mu_G*lambdaSigma-beta_x;
    GammaF=p.delta3^2*QF(p.gamma);
    Ctilde=p.delta3*m.iota/lt-GammaF-p.epsilon6*m.Leta^2- ...
        p.delta4*lv*m.normL;
    Dtilde=p.delta4*m.lambda2- ...
        p.epsilon4*p.delta3^2*m.lG^2*m.normL^2/(2*mv^2)- ...
        (p.delta4^2/2)*(p.epsilon4*m.Lx^2+p.epsilon6*m.Leta^2);
    Etilde=d_eta-p.epsilon5*c_eta- ...
        p.epsilon4*p.delta3^2*m.lG^2*m.Leta^2/(2*mv^2)-2/p.epsilon6;
    Schur=Btilde-(m.ell+m.tilde_ell)^2/(4*m.N*Atilde);

    cross=-(m.ell+m.tilde_ell)/(2*sqrt(m.N));
    M=zeros(5); M(1:2,1:2)=[Atilde,cross;cross,Btilde];
    M(3,3)=Ctilde; M(4,4)=Dtilde; M(5,5)=Etilde;
    minM=min(eig(M));

    canonical_curvature=(m.E-m.eta_half(1))*m.m_inner- ...
        m.eta_half(2)*m.rho2_curvature;

    cond={ ...
        'graph algebraic connectivity'; ...
        'uniform canonical Slater margin'; ...
        'target interior margin'; ...
        'target KKT construction margin'; ...
        'Delta subset int(E_plus): canonical curvature'; ...
        'negative original-constraint Hessian eigenvalue'; ...
        'Bregman sublevel r2 > W2(0)'; ...
        'regional l_theta <= 4.5'; ...
        'regional l_varphi <= 1.8'; ...
        'regional l_phi <= 1.8'; ...
        'core small gain: mu_c*d_eta > N*c_eta'; ...
        'epsilon5 interval'; ...
        'epsilon4 > 5N/(2mu5)'; ...
        'sigma admissible interval'; ...
        'epsilon6 > 2/e0'; ...
        'komega > bomega/lambda2'; ...
        'delta3 condition 1'; ...
        'delta3 condition 2'; ...
        'delta3 condition 3'; ...
        'delta4 = komega*delta3^2'; ...
        'varrho2 < 1/3'; ...
        'gamma > gammabar'; ...
        'iota > Eq.(40) threshold'; ...
        'A_tilde > 0'; ...
        'Schur margin > 0'; ...
        'C_tilde/delta3 > 0'; ...
        'D_tilde/delta3^2 > 0'; ...
        'E_tilde > 0'; ...
        'lambda_min(M_tilde) > 0'};

    val=[m.lambda2; m.slater_margin; m.target_interior_margin; ...
        m.target_kkt_margin; canonical_curvature; ...
        -m.nonconvex_hessian_min; cert.r2; ...
        cert.ltheta_exact; cert.lvarphi_exact; cert.lphi_exact; ...
        core_lhs; p.epsilon5; p.epsilon4; p.sigma; p.epsilon6; ...
        p.komega; d3_cond1; d3_cond2; d3_cond3; delta4_error; ...
        varrho2; p.gamma; m.iota; Atilde; Schur; ...
        Ctilde/p.delta3; Dtilde/p.delta3^2; Etilde; minM];

    lo=[0;0;0;0;m.mu_G;0;cert.W0; ...
        0;0;0;core_rhs;eps5_lo;eps4_min;0;eps6_min;komega_min; ...
        0;0;0;-inf;0;gammabar;iotareq;0;0;0;0;0;0];
    hi=[inf;inf;inf;inf;inf;inf;inf; ...
        m.ltheta_used;m.lvarphi_used;m.lphi_used;inf;eps5_hi;inf;sigma_max; ...
        inf;inf;inf;inf;inf;1e-24;1/3;inf;inf;inf;inf;inf;inf;inf;inf];

    pass=val>lo & val<hi;
    % Accept the equality check within floating-point tolerance.
    pass(20)=delta4_error<=1e-24;

    margin=min((val-lo)./max([abs(val),abs(lo),ones(size(val))*1e-14],[],2), ...
        (hi-val)./max([abs(val),abs(hi),ones(size(val))*1e-14],[],2));
    margin(20)=max(0,1-delta4_error/1e-24);
    finite_hi=isfinite(hi);
    infinite_hi=~finite_hi;
    margin(infinite_hi)=(val(infinite_hi)-lo(infinite_hi))./ ...
        max([abs(val(infinite_hi)),abs(lo(infinite_hi)), ...
        ones(sum(infinite_hi),1)*1e-14],[],2);

    T=table(string(cond),val,lo,hi,margin,pass, ...
        'VariableNames',{'Condition','Value','Lower','Upper','Margin','Pass'});

    fprintf(['Table-III certificate: core %.12e > %.12e; eps5=%.12e in ' ...
        '(%.12e, %.12e); eps4_min=%.12e; sigma_max=%.12e; ' ...
        'eps6_min=%.12e; komega_min=%.12e\n'], ...
        core_lhs,core_rhs,p.epsilon5,eps5_lo,eps5_hi,eps4_min,sigma_max, ...
        eps6_min,komega_min);
    fprintf(['delta3=%.12e, delta4=%.12e, varrho2=%.12e, gammabar=%.12e, ' ...
        'iota_req=%.12e\n'],p.delta3,p.delta4,varrho2,gammabar,iotareq);
    fprintf(['Lyapunov margins: A=%.12e, Schur=%.12e, C/d3=%.12e, ' ...
        'D/d3^2=%.12e, E=%.12e\n'], ...
        Atilde,Schur,Ctilde/p.delta3,Dtilde/p.delta3^2,Etilde);
end

function dy=proposed_rhs(y,m,p)
    N=m.N; n=m.n; nq=n*N*N; nw=N-1;
    Q=reshape(y(1:nq),n*N,N);
    nu=y(nq+(1:N));
    zeta=y(nq+N+(1:nw));
    omega=m.Romega*zeta;
    alpha_start=nq+N+nw;
    alpha=reshape(y(alpha_start+(1:2*N)),2,N);
    [X,x]=reconstruct(Q,m);
    lambda=fermi_map(nu,0,m.lambda_bar,m.mu_varphi);
    eta=zeros(2,N);
    for i=1:N
        eta(:,i)=fermi_map(alpha(:,i),m.etalo,m.etahi,m.mu_eta);
    end
    dQ=zeros(size(Q)); dnu=zeros(N,1); dalpha=zeros(2,N);
    for i=1:N
        rowsi=(i-1)*n+(1:n); xi=x(i,:).';
        Fi=m.q*(xi-m.r(i,:).');
        for j=find(m.A(i,:))
            rowsj=(j-1)*n+(1:n);
            Fi=Fi+m.beta*(xi-X(rowsj,i));
        end
        [rho,J]=rho_map(xi,i,m);
        own_cons=zeros(n,1);
        for j=find(m.A(i,:))
            own_cons=own_cons+(xi-X(rowsi,j));
        end
        dQ(rowsi,i)=-Fi-J.'*eta(:,i)*lambda(i)-p.gamma*own_cons;
        for player=1:N
            if player==i, continue; end
            rows=(player-1)*n+(1:n); v=zeros(n,1);
            for j=find(m.A(i,:))
                v=v+(X(rows,i)-X(rows,j));
            end
            dQ(rows,i)=-p.gamma*v;
        end
        gradstar=[(eta(1,i)-m.E)/m.a_outer;eta(2,i)/m.a_outer];
        phistar=((eta(1,i)-m.E)^2+eta(2,i)^2)/(2*m.a_outer) ...
            -m.local_offsets(i);
        dnu(i)=eta(:,i).'*rho-phistar-(m.L(i,:)*omega);
        dalpha(:,i)=lambda(i)*(rho-gradstar);
    end
    domega=m.L*lambda;
    dzeta=m.Romega_pinv*domega;
    dy=[dQ(:);dnu;dzeta;dalpha(:)];
end

function [X,x]=reconstruct(Q,m)
    X=Q; x=zeros(m.N,m.n);
    for i=1:m.N
        rows=(i-1)*m.n+(1:m.n);
        xi=fermi_map(Q(rows,i),m.xlo,m.xhi,m.mu_theta);
        X(rows,i)=xi; x(i,:)=xi.';
    end
end

function tr=decode_proposed(t,y,m)
    nt=numel(t); N=m.N; n=m.n; nq=n*N*N; nw=N-1;
    tr.t=t; tr.x=zeros(nt,N,n); tr.lambda=zeros(nt,N);
    tr.omega=zeros(nt,N); tr.eta=zeros(nt,N,2);
    alpha_start=nq+N+nw;
    tr.esterr=zeros(nt,1); tr.esterr_agent=zeros(nt,N); tr.stateerr=zeros(nt,1);
    tr.actionerr=zeros(nt,1);tr.lambdaerr=zeros(nt,1);tr.omegaerr=zeros(nt,1);
    tr.etaerr=zeros(nt,1);tr.totalerr=zeros(nt,1);
    tr.original_constraint=zeros(nt,1);tr.canonical_constraint=zeros(nt,1);
    tr.canonical_consistency=zeros(nt,1);
    tr.boundary_distance=zeros(nt,1);tr.primal_hessian_max=zeros(nt,1);
    xstar_stack=reshape(m.xstar.',[],1);
    Xstar=repmat(xstar_stack,1,N);
    for k=1:nt
        Q=reshape(y(k,1:nq),n*N,N); [X,x]=reconstruct(Q,m);
        tr.x(k,:,:)=reshape(x,1,N,n);
        la=fermi_map(y(k,nq+(1:N)).',0,m.lambda_bar,m.mu_varphi).';
        tr.lambda(k,:)=la;
        zeta=y(k,nq+N+(1:nw)).';
        tr.omega(k,:)=(m.Romega*zeta).';
        al=reshape(y(k,alpha_start+(1:2*N)),2,N);
        eta=zeros(2,N);
        for i=1:N
            eta(:,i)=fermi_map(al(:,i),m.etalo,m.etahi,m.mu_eta);
        end
        tr.eta(k,:,:)=reshape(eta.',1,N,2);
        go=0;gc=0;cc=0;
        for i=1:N
            [rho,~]=rho_map(x(i,:).',i,m);
            go=go+m.E*rho(1)+0.5*m.a_outer*(rho.'*rho)+m.local_offsets(i);
            phistar=((eta(1,i)-m.E)^2+eta(2,i)^2)/(2*m.a_outer) ...
                -m.local_offsets(i);
            gc=gc+eta(:,i).'*rho-phistar;
            etatrue=[m.E+m.a_outer*rho(1);m.a_outer*rho(2)];
            cc=cc+norm(eta(:,i)-etatrue)^2;
        end
        tr.original_constraint(k)=go; tr.canonical_constraint(k)=gc;
        tr.canonical_consistency(k)=sqrt(cc);
        xstack=reshape(x.',[],1);
        Xtruth=repmat(xstack,1,N);
        tr.esterr(k)=norm(X-Xtruth,'fro');
        for agent=1:N
            tr.esterr_agent(k,agent)=norm(X(:,agent)-xstack);
        end
        tr.actionerr(k)=norm(x-m.xstar,'fro');
        tr.lambdaerr(k)=norm(la-m.lambda_star);
        tr.omegaerr(k)=norm(tr.omega(k,:).'-m.omegastar);
        tr.etaerr(k)=norm(eta-m.etastar.','fro');
        tr.totalerr(k)=tr.actionerr(k)+tr.esterr(k)+tr.lambdaerr(k)+ ...
            tr.omegaerr(k)+tr.etaerr(k);
        tr.stateerr(k)=tr.totalerr(k);
        tr.boundary_distance(k)=min([ ...
            min(x-m.xlo.',[],'all'),min(m.xhi.'-x,[],'all')]);
        tr.primal_hessian_max(k)=max_fermi_hessian(x,m.xlo,m.xhi,m.mu_theta);
    end
end

function met=proposed_metrics(tr,m)
    met.normalized_error=tr.totalerr/max(tr.totalerr(1),realmin);
    met.log_error=log(max(met.normalized_error,1e-14));
    met.log_estimate=log(max(tr.esterr/max(tr.esterr(1),realmin),1e-14));
    met.consensus=max(abs(tr.lambda-mean(tr.lambda,2)),[],2);
    met.log_consensus=log(max(met.consensus/max(met.consensus(1),realmin),1e-14));
    met.omega_error=sqrt(sum((tr.omega-m.omegastar.').^2,2));
    met.log_omega_error=log(max(met.omega_error/max(met.omega_error(1),realmin),1e-14));
    met.log_eta=log(max(tr.etaerr/max(tr.etaerr(1),realmin),1e-14));
    met.log_consistency=log(max(tr.canonical_consistency/ ...
        max(tr.canonical_consistency(1),realmin),1e-14));
    met.log_canonical=met.log_consistency;
    idx=tr.t>=1.0 & met.normalized_error>1e-11;
    c=polyfit(tr.t(idx),met.log_error(idx),1);
    met.fit_rate=-c(1); met.log_fit=polyval(c,tr.t);
    xend=squeeze(tr.x(end,:,:)); laend=tr.lambda(end,:).';
    etaend=squeeze(tr.eta(end,:,:));
    R=zeros(m.N,m.n);
    for i=1:m.N
        [~,J]=rho_map(xend(i,:).',i,m);
        Fi=m.q*(xend(i,:).'-m.r(i,:).')+m.beta*(m.L(i,:)*xend).';
        R(i,:)=(Fi+J.'*etaend(i,:).'*laend(i)).';
    end
    met.kkt_terminal=norm(R,'fro');
end

function out=run_annular_globality_diagnostic()
    % Both flows use the same primal initialization and different dual states.
    d.N=2;
    d.q=[0.392;0.80136];
    d.beta=0;
    d.w=[1.00;1.35];
    d.rout=[0.98;1.08];
    d.rin=[0.42;0.55];
    d.a=0.5*(d.rout.^2+d.rin.^2);
    d.radial_halfwidth=0.5*(d.rout.^2-d.rin.^2);
    d.B=0.5*d.w.*(d.radial_halfwidth.^2);
    d.lambda_global=1.0;
    d.lambda_local=4.0;
    d.reference=d.rout + d.lambda_global*(2*d.w.*d.radial_halfwidth.*d.rout)./d.q;
    d.xlo=-2; d.xhi=2; d.etalo=0.05; d.etahi=1.0;
    d.kx=5; d.kdirect=10; d.klambda=10; d.keta=10;

    xlocal=-d.rin;
    xglobal=d.rout;
    x0=[xlocal(1)+0.015; xlocal(2)-0.020];
    t=linspace(0,20,1601).';
    opts=odeset('RelTol',2e-10,'AbsTol',2e-12,'MaxStep',0.005);
    yd0=[x0;d.lambda_local];
    [td,yd]=ode15s(@(~,y) annular_direct_rhs(y,d),t,yd0,opts);
    yc0=[x0;d.lambda_global;0.85*d.w.*d.radial_halfwidth];
    [tc,yc]=ode15s(@(~,y) annular_canonical_rhs(y,d),t,yc0,opts);
    assert(max(abs(td-t))<1e-12 && max(abs(tc-t))<1e-12);

    out.t=t;
    out.direct_x=yd(:,1:2); out.direct_lambda=max(yd(:,3),0);
    out.canonical_x=yc(:,1:2); out.canonical_lambda=max(yc(:,3),0);
    out.canonical_eta=min(max(yc(:,4:5),d.etalo),d.etahi);
    nt=numel(t);
    out.direct_gap=zeros(nt,1); out.canonical_gap=zeros(nt,1);
    out.direct_kkt=zeros(nt,1); out.canonical_kkt=zeros(nt,1);
    for kk=1:nt
        out.direct_gap(kk)=annular_global_gap(out.direct_x(kk,:).',d);
        out.canonical_gap(kk)=annular_global_gap(out.canonical_x(kk,:).',d);
        out.direct_kkt(kk)=annular_original_kkt(out.direct_x(kk,:).', ...
            out.direct_lambda(kk),d);
        out.canonical_kkt(kk)=annular_original_kkt(out.canonical_x(kk,:).', ...
            out.canonical_lambda(kk),d);
    end
    local_gap=annular_global_gap(xlocal,d);
    out.summary=table( ...
        ["direct original-constraint PD";"canonical lifted dynamics"], ...
        [out.direct_kkt(end);out.canonical_kkt(end)], ...
        [out.direct_gap(end);out.canonical_gap(end)], ...
        [annular_original_G(out.direct_x(end,:).',d); ...
         annular_original_G(out.canonical_x(end,:).',d)], ...
        [norm(out.direct_x(end,:).'-xlocal); ...
         norm(out.canonical_x(end,:).'-xglobal)], ...
        'VariableNames',{'Method','TerminalKKTResidual','TerminalGlobalBRGap', ...
        'TerminalConstraintResidual','DistanceToExpectedBranch'});
    out.model=d;
    fprintf('\nHETEROGENEOUS ANNULAR GLOBALITY DIAGNOSTIC\n');
    fprintf('Reference vector: [%.4f, %.4f].\n',d.reference(1),d.reference(2));
    fprintf('Expected nonglobal branch: [%.4f, %.4f].\n',xlocal(1),xlocal(2));
    fprintf('Expected global branch:   [%.4f, %.4f].\n',xglobal(1),xglobal(2));
    fprintf('Exact unilateral gap at the nonglobal branch: %.6f.\n',local_gap);
    fprintf('Direct terminal: KKT %.3e, gap %.6f, x=[%.6f %.6f].\n', ...
        out.direct_kkt(end),out.direct_gap(end), ...
        out.direct_x(end,1),out.direct_x(end,2));
    fprintf('Canonical terminal: KKT %.3e, gap %.3e, x=[%.6f %.6f].\n', ...
        out.canonical_kkt(end),out.canonical_gap(end), ...
        out.canonical_x(end,1),out.canonical_x(end,2));
    assert(out.direct_kkt(end)<2e-5 && out.direct_gap(end)>1, ...
        'The direct diagnostic did not reach the intended nonglobal KKT branch.');
    assert(out.canonical_gap(end)<1e-4, ...
        'The canonical diagnostic did not reach the global branch accurately.');
end

function dy=annular_direct_rhs(y,d)
    x=y(1:2); lambda=max(y(3),0);
    gradf=d.q.*(x-d.reference)+d.beta*[x(1)-x(2);x(2)-x(1)];
    gradg=2*d.w.*(x.^2-d.a).*x;
    dx=-(gradf+lambda*gradg);
    for i=1:2
        if x(i)<=d.xlo+1e-10 && dx(i)<0, dx(i)=0; end
        if x(i)>=d.xhi-1e-10 && dx(i)>0, dx(i)=0; end
    end
    dl=d.kdirect*annular_original_G(x,d);
    if lambda<=1e-12 && dl<0, dl=0; end
    dy=[dx;dl];
end

function dy=annular_canonical_rhs(y,d)
    x=y(1:2); lambda=max(y(3),0);
    eta=min(max(y(4:5),d.etalo),d.etahi);
    gradf=d.q.*(x-d.reference)+d.beta*[x(1)-x(2);x(2)-x(1)];
    dx=-d.kx*(gradf+2*lambda*eta.*x);
    for i=1:2
        if x(i)<=d.xlo+1e-10 && dx(i)<0, dx(i)=0; end
        if x(i)>=d.xhi-1e-10 && dx(i)>0, dx(i)=0; end
    end
    dl=d.klambda*annular_canonical_G(x,eta,d);
    if lambda<=1e-12 && dl<0, dl=0; end
    de=d.keta*lambda*(x.^2-d.a-eta./d.w);
    de(eta<=d.etalo+1e-10 & de<0)=0;
    de(eta>=d.etahi-1e-10 & de>0)=0;
    dy=[dx;dl;de];
end

function G=annular_original_G(x,d)
    G=sum(0.5*d.w.*(x.^2-d.a).^2-d.B);
end

function G=annular_canonical_G(x,eta,d)
    G=sum(eta.*(x.^2-d.a)-eta.^2./(2*d.w)-d.B);
end

function r=annular_original_kkt(x,lambda,d)
    gradf=d.q.*(x-d.reference)+d.beta*[x(1)-x(2);x(2)-x(1)];
    gradg=2*d.w.*(x.^2-d.a).*x;
    G=annular_original_G(x,d);
    r=norm(gradf+lambda*gradg)+max(G,0)+abs(lambda*G);
end

function gap=annular_global_gap(x,d)
    gaps=zeros(2,1);
    for i=1:2
        j=3-i;
        current=0.5*d.q(i)*(x(i)-d.reference(i))^2 + ...
            0.5*d.beta*(x(i)-x(j))^2;
        gj=0.5*d.w(j)*(x(j)^2-d.a(j))^2-d.B(j);
        rhs=d.B(i)-gj;
        if rhs<0
            gaps(i)=NaN; continue;
        end
        D=sqrt(max(0,2*rhs/d.w(i)));
        rlo=sqrt(max(0,d.a(i)-D));
        rhi=sqrt(max(0,d.a(i)+D));
        yfree=(d.q(i)*d.reference(i)+d.beta*x(j))/(d.q(i)+d.beta);
        candidates=[-rhi;-rlo;rlo;rhi; ...
            min(max(yfree,-rhi),-rlo);min(max(yfree,rlo),rhi)];
        vals=0.5*d.q(i)*(candidates-d.reference(i)).^2 + ...
            0.5*d.beta*(candidates-x(j)).^2;
        gaps(i)=max(current-min(vals),0);
    end
    gap=max(gaps,[],'omitnan');
end

function T=post_audit(pr,pm,m)
    names=["proposed terminal normalized total error"; ...
        "proposed terminal estimate disagreement"; ...
        "proposed multiplier disagreement"; ...
        "proposed active original constraint"; ...
        "proposed active canonical constraint"; ...
        "proposed canonical consistency"; ...
        "proposed terminal KKT residual"; ...
        "proposed positive fitted exponential rate"; ...
        "zero-sum omega invariant"; ...
        "terminal omega-target residual"; ...
        "negative original-constraint Hessian eigenvalue"];
    values=[pm.normalized_error(end);pr.esterr(end);pm.consensus(end); ...
        abs(pr.original_constraint(end));abs(pr.canonical_constraint(end)); ...
        pr.canonical_consistency(end);pm.kkt_terminal;pm.fit_rate; ...
        max(abs(sum(pr.omega,2)));pm.omega_error(end);-m.nonconvex_hessian_min];
    tolerances=[3e-5;3e-5;3e-5;3e-5;3e-5;2e-6;4e-2;0;2e-8;2e-5;0];
    passes=[values(1:7)<=tolerances(1:7);values(8)>0; ...
        values(9)<=tolerances(9);values(10)<=tolerances(10);values(11)>0];
    T=table(names,values,tolerances,passes, ...
        'VariableNames',{'Check','Value','Tolerance','Pass'});
end

function g=original_g(x,i,m)
    rho=rho_map(x,i,m);
    g=m.E*rho(1)+0.5*m.a_outer*(rho.'*rho)+m.local_offsets(i);
end

function [rho,J]=rho_map(x,i,m)
    u=x-m.xstar(i,:).';
    rho=[m.b*u(1)+0.5*m.m_inner*(u(1)^2+u(2)^2); ...
        m.rho2_amp*sin(m.wave*u(2))];
    J=[m.b+m.m_inner*u(1),m.m_inner*u(2); ...
        0,m.rho2_derivative_amp*cos(m.wave*u(2))];
end

function H=original_constraint_hessian(x,i,m)
    [rho,J]=rho_map(x,i,m);
    eta=[m.E+m.a_outer*rho(1);m.a_outer*rho(2)];
    u=x-m.xstar(i,:).';
    H=J.'*(m.a_outer*eye(2))*J+eta(1)*m.m_inner*eye(2);
    H(2,2)=H(2,2)+eta(2)*(-m.rho2_curvature*sin(m.wave*u(2)));
end

function G=aggregate_original_g(x,m)
    G=0;
    for i=1:m.N
        G=G+original_g(x(i,:).',i,m);
    end
end

function [g,gradg]=original_g_and_gradient(x,i,m)
    [rho,J]=rho_map(x,i,m);
    eta=[m.E+m.a_outer*rho(1);m.a_outer*rho(2)];
    g=m.E*rho(1)+0.5*m.a_outer*(rho.'*rho)+m.local_offsets(i);
    gradg=J.'*eta;
end

function dmin=min_pairwise_distance(X)
    dmin=inf;
    for i=1:size(X,1)-1
        for j=i+1:size(X,1)
            dmin=min(dmin,norm(X(i,:)-X(j,:)));
        end
    end
end

function Hmax=max_fermi_hessian(x,lo,hi,mu)
    lo=lo(:).';hi=hi(:).';mu=expand(mu,size(x,2)).';
    width=hi-lo;kk=mu.*width/4;
    H=kk.*width./((x-lo).*(hi-x));
    Hmax=max(H,[],'all');
end

function make_topology(m,out_dir)
    f=figure('Color','w','Position',[100 100 620 520]);
    p=plot(graph(m.A),'Layout','circle','LineWidth',1.6, ...
        'NodeColor',[0.15 0.36 0.62],'EdgeColor',[0.42 0.42 0.42], ...
        'MarkerSize',9,'NodeLabel',1:m.N);
    p.NodeFontSize=11;
    axis off;
    save_figure_bundle(f,out_dir,'shared_topology');
    close(f);
end

function make_figures(pr,pm,m,cert,out_dir) %#ok<INUSD>
    cols=agent_colors();
    blue=cols(1,:);
    red=cols(2,:);
    nt=numel(pr.t);

    aggregate_action_error=zeros(nt,1);
    for i=1:m.N
        dx=squeeze(pr.x(:,i,:))-m.xstar(i,:);
        aggregate_action_error=aggregate_action_error+sum(dx.^2,2);
    end
    aggregate_estimate_error=sum(pr.esterr_agent.^2,2);

    % Primal and decision-estimate errors
    f=figure('Color','w','Position',[100 100 1260 480]);
    tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');

    ax=nexttile(tl); hold(ax,'on');
    semilogy(ax,pr.t,max(aggregate_action_error,1e-16), ...
        'Color',blue,'LineWidth',2.0, ...
        'DisplayName','$\sum_{i=1}^{N}\|x_i-x_i^\circ\|^2$');
    grid(ax,'on');
    xlabel(ax,'$t$','Interpreter','latex');
    ylabel(ax,'$\sum_{i=1}^{N}\|x_i(t)-x_i^\circ\|^2$', ...
        'Interpreter','latex');
    legend(ax,'Location','best','Interpreter','latex');

    ax=nexttile(tl); hold(ax,'on');
    semilogy(ax,pr.t,max(aggregate_estimate_error,1e-16), ...
        'Color',red,'LineWidth',2.0, ...
        'DisplayName','$\sum_{i=1}^{N}\|\mathbf{x}^i-\mathrm{col}\{x_j\}\|^2$');
    grid(ax,'on');
    xlabel(ax,'$t$','Interpreter','latex');
    ylabel(ax,['$\sum_{i=1}^{N}\|\mathbf{x}^i(t)-' ...
        '\mathrm{col}\{x_j(t)\}_{j=1}^{N}\|^2$'], ...
        'Interpreter','latex');
    legend(ax,'Location','best','Interpreter','latex');

    format_axes(f);
    save_figure_bundle(f,out_dir,'nonconvex_tra');
    close(f);

    % Multiplier deviations and auxiliary states
    f=figure('Color','w','Position',[80 100 1460 500]);
    tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');

    ax=nexttile(tl); hold(ax,'on');
    hlam=gobjects(m.N,1);
    lambda_err=1e4*(pr.lambda-m.lambda_star);
    for i=1:m.N
        hlam(i)=plot(ax,pr.t,lambda_err(:,i),'Color',cols(i,:), ...
            'LineWidth',5,'DisplayName',sprintf('Node %d',i));
    end
    href=yline(ax,0,'k--','LineWidth',2.0, ...
        'DisplayName','$\lambda_i=\tilde{\lambda}^{\circ}$');
    grid(ax,'on');
    xlabel(ax,'$t$','Interpreter','latex');
    ylabel(ax,'$10^4(\lambda_i-\tilde{\lambda}^{\circ})$', ...
        'Interpreter','latex');
    legend(ax,[hlam;href],'Location','eastoutside','Interpreter','latex');

    ax=nexttile(tl); hold(ax,'on');
    homega=gobjects(m.N,1);
    for i=1:m.N
        homega(i)=plot(ax,pr.t,pr.omega(:,i),'Color',cols(i,:), ...
            'LineWidth',5,'DisplayName',sprintf('Node %d',i));
    end
    hsum=plot(ax,pr.t,sum(pr.omega,2),'k--','LineWidth',2.0, ...
        'DisplayName','$\sum_i\omega_i(t)$');
    grid(ax,'on');
    xlabel(ax,'$t$','Interpreter','latex');
    ylabel(ax,'$\omega_i$','Interpreter','latex');
    legend(ax,[homega;hsum],'Location','eastoutside','Interpreter','latex');

    format_axes(f);
    save_figure_bundle(f,out_dir,'nonconvex_aux');
    close(f);

    % Convergence and constraint residuals
    f=figure('Color','w','Position',[100 100 1260 480]);
    tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');

    ax=nexttile(tl); hold(ax,'on');
    green=cols(3,:);
    h1=plot(ax,pr.t,pm.log_error,'LineWidth',2.0,'Color',blue, ...
        'DisplayName','$e_{\Sigma,2}$');
    h2=plot(ax,pr.t,pm.log_eta,'LineWidth',1.8,'Color',red, ...
        'DisplayName','$e_{\eta}=\|\eta-\eta^{\circ}\|_{F}$');
    h3=plot(ax,pr.t,pm.log_consistency,'LineWidth',1.8,'Color',green, ...
        'DisplayName','$e_{\rm c}=\|\eta-\nabla\Phi(\rho(x))\|_{F}$');
    grid(ax,'on');
    xlabel(ax,'Time $t$ (s)','Interpreter','latex');
    ylabel(ax,'$\ln(e(t)/e(0))$','Interpreter','latex');
    xlim(ax,[0,max(pr.t)]);
    legend(ax,[h1 h2 h3],'Location','southwest','Interpreter','latex');

    ax=nexttile(tl); hold(ax,'on');
    h1=plot(ax,pr.t,pr.original_constraint,'LineWidth',2.0,'Color',blue, ...
        'DisplayName','original $G(x)$');
    h2=plot(ax,pr.t,pr.canonical_constraint,'--','LineWidth',1.55, ...
        'Color',red,'DisplayName','canonical $\tilde{G}(x,\eta)$');
    yline(ax,0,'k:','LineWidth',1.20,'HandleVisibility','off');
    grid(ax,'on');
    xlabel(ax,'Time $t$ (s)','Interpreter','latex');
    ylabel(ax,'Constraint residual','Interpreter','latex');
    xlim(ax,[0,1e-3]);
    legend(ax,[h1 h2],'Location','best','Interpreter','latex');

    format_axes(f);
    save_figure_bundle(f,out_dir,'nonconvex_exp');
    close(f);
end

function make_globality_diagnostic_figures(diagnostic,out_dir)
    cols=agent_colors();
    blue=cols(1,:);
    red=cols(2,:);
    d=diagnostic.model;

    f=figure('Color','w','Position',[80 100 1420 500]);
    tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');

    ax=nexttile(tl); hold(ax,'on');
    h1=plot(ax,diagnostic.t,diagnostic.direct_x(:,1),'--','LineWidth',1.8, ...
        'Color',blue,'DisplayName','direct PD: player 1');
    h2=plot(ax,diagnostic.t,diagnostic.direct_x(:,2),':','LineWidth',2.0, ...
        'Color',blue,'DisplayName','direct PD: player 2');
    h3=plot(ax,diagnostic.t,diagnostic.canonical_x(:,1),'-','LineWidth',1.9, ...
        'Color',red,'DisplayName','canonical: player 1');
    h4=plot(ax,diagnostic.t,diagnostic.canonical_x(:,2),'-.','LineWidth',2.0, ...
        'Color',red,'DisplayName','canonical: player 2');
    h5=yline(ax,-d.rin(1),'Color',blue,'LineStyle',':','LineWidth',1.0, ...
        'DisplayName','player 1 nonglobal branch');
    h6=yline(ax,-d.rin(2),'Color',blue,'LineStyle','-.','LineWidth',1.0, ...
        'DisplayName','player 2 nonglobal branch');
    h7=yline(ax,d.rout(1),'Color',red,'LineStyle','--','LineWidth',1.1, ...
        'DisplayName','player 1 global branch');
    h8=yline(ax,d.rout(2),'Color',red,'LineStyle',':','LineWidth',1.1, ...
        'DisplayName','player 2 global branch');
    grid(ax,'on');
    xlim(ax,[0,3]);
    ylim(ax,[-0.70,1.20]);
    xlabel(ax,'$t$','Interpreter','latex');
    ylabel(ax,'$x_i$','Interpreter','latex');
    legend(ax,[h1 h2 h3 h4 h5 h6 h7 h8], ...
        'Location','eastoutside','Interpreter','latex');

    ax=nexttile(tl); hold(ax,'on');
    h1=semilogy(ax,diagnostic.t,max(diagnostic.direct_gap,1e-12), ...
        '--','LineWidth',1.9,'Color',blue, ...
        'DisplayName','direct original-constraint PD');
    h2=semilogy(ax,diagnostic.t,max(diagnostic.canonical_gap,1e-12), ...
        '-','LineWidth',2.0,'Color',red, ...
        'DisplayName','canonical lifted dynamics');
    grid(ax,'on');
    xlabel(ax,'$t$','Interpreter','latex');
    ylabel(ax,'$\mathcal{G}_{\mathrm{BR}}(x(t))$','Interpreter','latex');
    legend(ax,[h1 h2],'Location','best','Interpreter','latex');

    format_axes(f);
    save_figure_bundle(f,out_dir,'nonconvex_globality');
    close(f);
end

function write_globality_report(diagnostic,out_dir)
    file=fullfile(out_dir,'globality_gap_report.txt');
    fid=fopen(file,'w');
    assert(fid>=0,'Cannot create globality report.');
    cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
    d=diagnostic.model;
    fprintf(fid,'SEPARATE STATIONARY-VERSUS-GLOBAL DIAGNOSTIC\n');
    fprintf(fid,'Two-player annular example, separate from the ten-agent certificate.\n\n');
    fprintf(fid,['Annular game parameters:\n' ...
        'Player 1 uses r_in=%.2f, r_out=%.2f, q_1=%.5f.\n' ...
        'Player 2 uses r_in=%.2f, r_out=%.2f, q_2=%.5f.\n' ...
        '\n'], ...
        d.rin(1),d.rout(1),d.q(1),d.rin(2),d.rout(2),d.q(2));
    fprintf(fid,['Both flows use the same initial primal profile, with different ' ...
        'multiplier and canonical-state initializations.\n\n']);
    fprintf(fid,'Direct terminal KKT residual : %.12e\n',diagnostic.direct_kkt(end));
    fprintf(fid,'Direct terminal global BR gap: %.12e\n',diagnostic.direct_gap(end));
    fprintf(fid,'Canonical terminal KKT residual : %.12e\n',diagnostic.canonical_kkt(end));
    fprintf(fid,'Canonical terminal global BR gap: %.12e\n',diagnostic.canonical_gap(end));
    fprintf(fid,['\nA small KKT residual with a positive unilateral global best-response ' ...
        'gap certifies stationarity without global Nash optimality.\n']);
    fprintf(fid,['This diagnostic is separate from ' ...
        'the ten-agent theorem-parameter certificate.\n']);
end

function cols=agent_colors()
    cols=[ ...
        0.00, 0.00, 1.00; ...
        1.00, 0.00, 0.00; ...
        0.00, 0.50, 0.00; ...
        1.00, 0.50, 0.00; ...
        0.50, 0.00, 1.00; ...
        0.00, 0.75, 1.00; ...
        1.00, 0.00, 0.50; ...
        0.60, 0.30, 0.00; ...
        0.50, 0.50, 0.00; ...
        0.25, 0.25, 0.25];
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

function x=fermi_map(z,lo,hi,mu)
    z=z(:);lo=expand(lo,numel(z));hi=expand(hi,numel(z));mu=expand(mu,numel(z));
    width=hi-lo;kk=mu.*width/4;a=z./kk;s=zeros(size(z));pos=a>=0;
    s(pos)=1./(1+exp(-min(a(pos),700)));
    ea=exp(max(a(~pos),-700));s(~pos)=ea./(1+ea);x=lo+width.*s;
end

function z=fermi_inverse(x,lo,hi,mu)
    x=x(:);lo=expand(lo,numel(x));hi=expand(hi,numel(x));mu=expand(mu,numel(x));
    width=hi-lo;kk=mu.*width/4;p=min(max((x-lo)./width,1e-14),1-1e-14);
    z=kk.*log(p./(1-p));
end

function D=fermi_D(ref,current,lo,hi,mu)
    ref=ref(:);current=current(:);lo=expand(lo,numel(ref));hi=expand(hi,numel(ref));
    mu=expand(mu,numel(ref));width=hi-lo;kk=mu.*width/4;
    pr=min(max((ref-lo)./width,1e-14),1-1e-14);
    pc=min(max((current-lo)./width,1e-14),1-1e-14);
    D=kk.*width.*(pr.*log(pr./pc)+(1-pr).*log((1-pr)./(1-pc)));
end

function c=fermi_level_curvature(ref,lo,hi,mu,level)
    % Curvature on the full Bregman sublevel.
    % D = (mu*width^2/4)*KL(p_ref || p); psi''(x) = mu/[4*p*(1-p)].
    width=hi-lo;
    pref=min(max((ref-lo)/width,1e-12),1-1e-12);
    scale=mu*width^2/4;

    Dp=@(p) scale*(pref*log(pref./p)+ ...
        (1-pref)*log((1-pref)./(1-p)))-level;

    left_hi=max(pref-1e-12,2e-14);
    right_lo=min(pref+1e-12,1-2e-14);

    if Dp(1e-14)<=0 || Dp(1-1e-14)<=0
        error(['The requested Bregman sublevel reaches too close to a ' ...
            'Fermi--Dirac boundary for a finite numerical curvature ' ...
            'certificate at the current floating-point resolution.']);
    end

    pleft=fzero(Dp,[1e-14,left_hi]);
    pright=fzero(Dp,[right_lo,1-1e-14]);
    c=max(mu/(4*pleft*(1-pleft)), ...
          mu/(4*pright*(1-pright)));
end

function v=expand(v,n)
    if isscalar(v),v=repmat(v,n,1);else,v=v(:);end
end
