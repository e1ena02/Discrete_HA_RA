function model = solve_EGP(p, grids, heterogeneity,...
    income, futureshock, periods_until_shock, nextmodel)
    % This function performs the method of endogenous grid points to find
    % saving and consumption policy functions. It also calls 
    % find_stationary() to compute the stationary distribution over states 
    % via direct methods (rather than simulations) and stores the results 
    % in the 'model' structure.
    %
    % To compute the MPCs out of news, it is necessary for the policy function
    % to reflect the expectation of a future shock. For these cases,
    % the policy functions in 'nextmodel' are used. The variable 'nextmpcshock'
    % is nonzero when a shock is expected next period.
    %
    % Brian Livingston, 2020
    % livingstonb@uchicago.edu

    %% ----------------------------------------------------
    % USEFUL OBJECTS/ARRAYS
    % -----------------------------------------------------
    nextmpcshock = (periods_until_shock == 1) * futureshock;

    ss_dims = [p.nx, p.nyP, p.nyF, p.nz];
    R_bc = heterogeneity.R_broadcast;

    % If expected future shock is negative, need to raise today's
    % borrowing limit to satisfy future budget constraints with prob one
    % (in all other cases, it should hold that adj_borr_lims = p.borrow_lim).
    tmp = p.borrow_lim - futureshock - income.minnety;
    adj_borr_lims = max(tmp ./ reshape(p.R, 1, []), p.borrow_lim);

    adj_borr_lims_bc = adj_borr_lims(:);
    if numel(adj_borr_lims_bc) == 1
        adj_borr_lims_bc = repmat(adj_borr_lims_bc, p.nz, 1);
    end
    adj_borr_lims_bc = shiftdim(adj_borr_lims_bc, -3);

    svecs_bc = grids.s.vec + (adj_borr_lims_bc - p.borrow_lim);
    svecs_bc_tax = p.compute_savtax(svecs_bc);
    
    xmat = grids.x.matrix + R_bc .* (adj_borr_lims_bc - p.borrow_lim);

    tempt_expr = heterogeneity.temptation_broadcast;
    tempt_expr = tempt_expr ./ (1 + tempt_expr);

    % Find xprime as a function of s
    xprime_s = R_bc .* svecs_bc + income.netymatEGP + nextmpcshock;

    % Extend to full state space
    newdims = [ss_dims p.nyT];
    [n1, n2, n3, n4, n5] = size(xprime_s);
    dims0 = [n1, n2, n3, n4, n5];
    newdims(newdims == dims0) = 1;
    xprime_s = repmat(xprime_s, newdims);

    %% ----------------------------------------------------
    % CONSTRUCT EXPECTATIONS MATRIX, ETC...
    % -----------------------------------------------------
    Emat = kron(income.ytrans_live, speye(p.nx));

    % Initial guess for consumption function
    tempt_adj = 0.5 * (max(p.temptation) > 0.05);
    con = (max(heterogeneity.r_broadcast, 0.001) + tempt_adj) .* xmat;
    con = con(:);
    con(con<=0) = min(con(con>0));
    con = reshape(con, ss_dims);

    %% ----------------------------------------------------
    % EGP ITERATION
    % ----------------------------------------------------- 
    iter = 1;
    cdiff = 1;
    conlast = con;
    while iter<p.max_iter && cdiff>p.tol_iter
        if iter > 1
            conlast = conupdate;
        end
        iter = iter + 1;

        % interpolate to get c(x') using c(x)
         
        % c(x')
        c_xp = get_c_xprime(p, grids, xprime_s, ...
            nextmodel, conlast, nextmpcshock, xmat);
        if p.phi ~= 0
            c_xp = c_xp + p.phi .* c_xp;
        end

        % MUC in current period, from Euler equation
        muc_s = get_marginal_util_cons(...
            p, income, grids, c_xp, xprime_s, R_bc,...
            Emat, heterogeneity.betagrid_broadcast,...
            heterogeneity.risk_aver_broadcast,...
            tempt_expr, svecs_bc);
         % c(s)
        con_s = aux.u1inv(heterogeneity.risk_aver_broadcast, muc_s);

        if p.phi ~= 0
            muc_s = muc_s + p.phi .* muc_s;
            con_s = aux.u1inv(heterogeneity.risk_aver_broadcast, muc_s);
        end
        
        % x(s) = s + stax + c(s)
        x_s = svecs_bc + svecs_bc_tax + con_s;

        % interpolate from x(s) to get s(x)
        sav = get_saving_policy(p, grids, x_s, nextmpcshock,...
            R_bc, svecs_bc, xmat);
        sav_tax = p.compute_savtax(sav);
       
        % updated consumption function, column vec length of
        conupdate = xmat - sav - sav_tax;

        cdiff = max(abs(conupdate(:)-conlast(:)));
        if (mod(iter, 100) == 0) && ~p.calibrating
            disp(['  EGP Iteration ' int2str(iter), ' max con fn diff is ' num2str(cdiff)]);
        end
    end

    if cdiff>p.tol_iter
        % EGP did not converge, don't find stationary distribution
        AYdiff = 100000;
        model.EGP_cdiff = cdiff;
        return
    end

    model.sav = sav;
    model.con = conupdate;
    model.EGP_cdiff = cdiff;

    % create interpolants from optimal policy functions
    % and find saving values associated with xvals
    % max_sav = (p.borrow_lim - min(income.netymat(:)) - nextmpcshock) ./ p.R;

    model.savinterp = cell(p.nyP,p.nyF,p.nz);
    model.coninterp = cell(p.nyP,p.nyF,p.nz);
    model.coninterp_ext = cell(p.nyP,p.nyF,p.nz);
    model.coninterp_mpc = cell(p.nyP,p.nyF,p.nz);
    for ib = 1:p.nz
    for iyF = 1:p.nyF
    for iyP = 1:p.nyP
        model.savinterp{iyP,iyF,ib} = griddedInterpolant(...
            xmat(:,iyP,iyF,ib), model.sav(:,iyP,iyF,ib), 'linear');

        model.coninterp{iyP,iyF,ib} = griddedInterpolant(...
            xmat(:,iyP,iyF,ib), model.con(:,iyP,iyF,ib), 'linear');

        xmin = xmat(1,iyP,iyF,ib);
        cmin = model.con(1,iyP,iyF,ib);
        lbound = 1e-7;

        % Create adjusted interpolants for negative shocks
        model.coninterp_ext{iyP,iyF,ib} = @(x) extend_interp(...
            model.coninterp{iyP,iyF,ib}, x, xmin, cmin, lbound,...
            adj_borr_lims_bc(1,1,1,ib));

        model.coninterp_mpc{iyP,iyF,ib} = @(x) extend_interp(...
            model.coninterp{iyP,iyF,ib}, x, xmin, cmin, -inf,...
            adj_borr_lims_bc(1,1,1,ib));
    end
    end
    end

    model.adj_borr_lims = adj_borr_lims;
    model.adj_borr_lims_bc = adj_borr_lims_bc;
end

function out = extend_interp(old_interpolant, qvals, gridmin,...
    valmin, lb, blim)
    % Applies a correction to the interpolating function since negative
    % shock may push wealth below borrowing limit (only relevant for
    % computing MPCs out of negative shocks)

    out = zeros(size(qvals));
    adj = qvals < gridmin;
    out(~adj) = old_interpolant(qvals(~adj));
    out(adj) = valmin + qvals(adj) - gridmin;
    out(adj) = min(out(adj), qvals(adj)-blim);
    out = max(out, lb);
end

function c_xprime = get_c_xprime(p, grids, xp_s, nextmodel, conlast,...
    nextmpcshock, xmat)
	% find c as a function of x'
	c_xprime = zeros(size(xp_s));
    dims_nx_nyT = [p.nx, 1, 1, 1, p.nyT];
	for ib  = 1:p.nz
    for iyF = 1:p.nyF
    for iyP = 1:p.nyP
    	xp_s_ib_iyF_iyP = xp_s(:,iyP,iyF,ib,:);

        if isempty(nextmodel)
            % Usual method of EGP
            coninterp = griddedInterpolant(...
                xmat(:,iyP,iyF,ib), conlast(:,iyP,iyF,ib), 'linear');
            c_xprime(:,iyP,iyF,ib,:) = reshape(...
                coninterp(xp_s_ib_iyF_iyP(:)), dims_nx_nyT);
        else
            % Solve conditional on an expected future shock using
            % next period's policy functions
            c_xprime(:,iyP,iyF,ib,:) = reshape(...
                nextmodel.coninterp_ext{iyP,iyF,ib}(xp_s_ib_iyF_iyP(:)), dims_nx_nyT);
        end
    end
    end
    end
end

function muc_s = get_marginal_util_cons(...
	p, income, grids, c_xp, xp_s, R_bc,...
    Emat, betagrid_bc, risk_aver_bc, tempt_expr,...
    svecs_bc)

    savtaxrate = (1 + p.savtax .* (svecs_bc >= p.savtaxthresh));

	% First get marginal utility of consumption next period
	muc_c = aux.utility1(risk_aver_bc, c_xp);
    muc_tempt = -tempt_expr .* aux.utility1(risk_aver_bc, xp_s+1e-7);
    mucnext = reshape(muc_c(:) + muc_tempt(:), [], p.nyT);

    % Integrate
    expectation = Emat * mucnext * income.yTdist;
    expectation = reshape(expectation, [p.nx, p.nyP, p.nyF, p.nz]);

    muc_c_today = R_bc .* betagrid_bc .* expectation;
    muc_beq = aux.utility_bequests1(p.bequest_curv, p.bequest_weight,...
        p.bequest_luxury, svecs_bc);

    muc_s = (1 - p.dieprob) * muc_c_today ./ savtaxrate ...
        + p.dieprob * muc_beq;
end

function sav = get_saving_policy(p, grids, x_s, nextmpcshock, R_bc,...
    svecs_bc, xmat)
	% Finds s(x), the saving policy function on the
	% cash-on-hand grid

    sav = zeros(size(x_s));
    xstar = zeros(p.nyP,p.nyF,p.nz);
    for ib  = 1:p.nz
    for iyF = 1:p.nyF
    for iyP = 1:p.nyP
        adj = xmat(:,iyP,iyF,ib) < x_s(1,iyP,iyF,ib);
        sav(adj,iyP,iyF,ib) = svecs_bc(1,1,1,ib);

        savinterp = griddedInterpolant(x_s(:,iyP,iyF,ib),...
            svecs_bc(:,1,1,ib), 'linear');

        sav(~adj,iyP,iyF,ib) = savinterp(...
            xmat(~adj,iyP,iyF,ib));
    end
    end
    end
end
% function model = solve_EGP(p, grids, heterogeneity, income, futureshock, periods_until_shock, nextmodel)
%     if p.phi == 0
%     % This function performs the method of endogenous grid points to find
%     % saving and consumption policy functions. It also calls 
%     % find_stationary() to compute the stationary distribution over states 
%     % via direct methods (rather than simulations) and stores the results 
%     % in the 'model' structure.
%     %
%     % To compute the MPCs out of news, it is necessary for the policy function
%     % to reflect the expectation of a future shock. For these cases,
%     % the policy functions in 'nextmodel' are used. The variable 'nextmpcshock'
%     % is nonzero when a shock is expected next period.
%     %
%     % Brian Livingston, 2020
%     % livingstonb@uchicago.edu
% 
%     %% ----------------------------------------------------
%     % USEFUL OBJECTS/ARRAYS
%     % -----------------------------------------------------
%     nextmpcshock = (periods_until_shock == 1) * futureshock;
% 
%     ss_dims = [p.nx, p.nyP, p.nyF, p.nz];
%     R_bc = heterogeneity.R_broadcast;
% 
%     % If expected future shock is negative, need to raise today's
%     % borrowing limit to satisfy future budget constraints with prob one
%     % (in all other cases, it should hold that adj_borr_lims = p.borrow_lim).
%     tmp = p.borrow_lim - futureshock - income.minnety;
%     adj_borr_lims = max(tmp ./ reshape(p.R, 1, []), p.borrow_lim);
% 
%     adj_borr_lims_bc = adj_borr_lims(:);
%     if numel(adj_borr_lims_bc) == 1
%         adj_borr_lims_bc = repmat(adj_borr_lims_bc, p.nz, 1);
%     end
%     adj_borr_lims_bc = shiftdim(adj_borr_lims_bc, -3);
% 
%     svecs_bc = grids.s.vec + (adj_borr_lims_bc - p.borrow_lim);
%     svecs_bc_tax = p.compute_savtax(svecs_bc);
% 
%     xmat = grids.x.matrix + R_bc .* (adj_borr_lims_bc - p.borrow_lim);
% 
%     tempt_expr = heterogeneity.temptation_broadcast;
%     tempt_expr = tempt_expr ./ (1 + tempt_expr);
% 
%     % Find xprime as a function of s
%     xprime_s = R_bc .* svecs_bc + income.netymatEGP + nextmpcshock;
% 
%     % Extend to full state space
%     newdims = [ss_dims p.nyT];
%     [n1, n2, n3, n4, n5] = size(xprime_s);
%     dims0 = [n1, n2, n3, n4, n5];
%     newdims(newdims == dims0) = 1;
%     xprime_s = repmat(xprime_s, newdims);
% 
%     %% ----------------------------------------------------
%     % CONSTRUCT EXPECTATIONS MATRIX, ETC...
%     % -----------------------------------------------------
%     Emat = kron(income.ytrans_live, speye(p.nx));
% 
%     % Initial guess for consumption function
%     tempt_adj = 0.5 * (max(p.temptation) > 0.05);
%     con = (max(heterogeneity.r_broadcast, 0.001) + tempt_adj) .* xmat;
%     con = con(:);
%     con(con<=0) = min(con(con>0));
%     con = reshape(con, ss_dims);
% 
%     %% ----------------------------------------------------
%     % EGP ITERATION
%     % ----------------------------------------------------- 
%     iter = 1;
%     cdiff = 1;
%     conlast = con;
%     while iter<p.max_iter && cdiff>p.tol_iter
%         if iter > 1
%             conlast = conupdate;
%         end
%         iter = iter + 1;
% 
%         % interpolate to get c(x') using c(x)
% 
%         % c(x')
%         c_xp = get_c_xprime(p, grids, xprime_s, ...
%             nextmodel, conlast, nextmpcshock, xmat);
%         assignin('base','c_xp', c_xp);
% 
%         % MUC in current period, from Euler equation
%         muc_s = get_marginal_util_cons(...
%             p, income, grids, c_xp, xprime_s, R_bc,...
%             Emat, heterogeneity.betagrid_broadcast,...
%             heterogeneity.risk_aver_broadcast,...
%             tempt_expr, svecs_bc);
%         assignin('base', 'muc_s', muc_s);
% 
%         % c(s)
%         con_s = aux.u1inv(heterogeneity.risk_aver_broadcast, muc_s);
%         assignin('base', 'con_s', con_s);
%         % x(s) = s + stax + c(s)
%         x_s = svecs_bc + svecs_bc_tax + con_s;
% 
%         % interpolate from x(s) to get s(x)
%         sav = get_saving_policy(p, grids, x_s, nextmpcshock,...
%             R_bc, svecs_bc, xmat);
%         sav_tax = p.compute_savtax(sav);
%         assignin('base', 'sav', sav);
%         foo()
%         % updated consumption function, column vec length of
%         conupdate = xmat - sav - sav_tax;
% 
%         cdiff = max(abs(conupdate(:)-conlast(:)));
%         if (mod(iter, 100) == 0) && ~p.calibrating
%             disp(['  EGP Iteration ' int2str(iter), ' max con fn diff is ' num2str(cdiff)]);
%         end
%     end
% 
%     if cdiff>p.tol_iter
%         % EGP did not converge, don't find stationary distribution
%         AYdiff = 100000;
%         model.EGP_cdiff = cdiff;
%         return
%     end
% 
%     model.sav = sav;
%     model.con = conupdate;
%     model.EGP_cdiff = cdiff;
% 
%     % create interpolants from optimal policy functions
%     % and find saving values associated with xvals
%     % max_sav = (p.borrow_lim - min(income.netymat(:)) - nextmpcshock) ./ p.R;
% 
%     model.savinterp = cell(p.nyP,p.nyF,p.nz);
%     model.coninterp = cell(p.nyP,p.nyF,p.nz);
%     model.coninterp_ext = cell(p.nyP,p.nyF,p.nz);
%     model.coninterp_mpc = cell(p.nyP,p.nyF,p.nz);
%     for ib = 1:p.nz
%     for iyF = 1:p.nyF
%     for iyP = 1:p.nyP
%         model.savinterp{iyP,iyF,ib} = griddedInterpolant(...
%             xmat(:,iyP,iyF,ib), model.sav(:,iyP,iyF,ib), 'linear');
% 
%         model.coninterp{iyP,iyF,ib} = griddedInterpolant(...
%             xmat(:,iyP,iyF,ib), model.con(:,iyP,iyF,ib), 'linear');
% 
%         xmin = xmat(1,iyP,iyF,ib);
%         cmin = model.con(1,iyP,iyF,ib);
%         lbound = 1e-7;
% 
%         % Create adjusted interpolants for negative shocks
%         model.coninterp_ext{iyP,iyF,ib} = @(x) extend_interp(...
%             model.coninterp{iyP,iyF,ib}, x, xmin, cmin, lbound,...
%             adj_borr_lims_bc(1,1,1,ib));
% 
%         model.coninterp_mpc{iyP,iyF,ib} = @(x) extend_interp(...
%             model.coninterp{iyP,iyF,ib}, x, xmin, cmin, -inf,...
%             adj_borr_lims_bc(1,1,1,ib));
%     end
%     end
%     end
% 
%     model.adj_borr_lims = adj_borr_lims;
%     model.adj_borr_lims_bc = adj_borr_lims_bc;
%     end
% else
%     nextmpcshock = (periods_until_shock == 1) * futureshock;
% 
%     ss_dims = [p.nx, p.nyP, p.nyF, p.nz];
%     R_bc = heterogeneity.R_broadcast;
% 
%     % If expected future shock is negative, need to raise today's
%     % borrowing limit to satisfy future budget constraints with prob one
%     % (in all other cases, it should hold that adj_borr_lims = p.borrow_lim).
%     tmp = p.borrow_lim - futureshock - income.minnety;
%     adj_borr_lims = max(tmp ./ reshape(p.R, 1, []), p.borrow_lim);
% 
%     adj_borr_lims_bc = adj_borr_lims(:);
%     if numel(adj_borr_lims_bc) == 1
%         adj_borr_lims_bc = repmat(adj_borr_lims_bc, p.nz, 1);
%     end
%     adj_borr_lims_bc = shiftdim(adj_borr_lims_bc, -3);
% 
%     svecs_bc = grids.s.vec + (adj_borr_lims_bc - p.borrow_lim);
%     svecs_bc_tax = p.compute_savtax(svecs_bc);
% 
%     xmat = grids.x.matrix + R_bc .* (adj_borr_lims_bc - p.borrow_lim);
% 
%     tempt_expr = heterogeneity.temptation_broadcast;
%     tempt_expr = tempt_expr ./ (1 + tempt_expr);
% 
%     % Find xprime as a function of s
%     xprime_s = R_bc .* svecs_bc + income.netymatEGP + nextmpcshock;
% 
%     % Extend to full state space
%     newdims = [ss_dims p.nyT];
%     [n1, n2, n3, n4, n5] = size(xprime_s);
%     dims0 = [n1, n2, n3, n4, n5];
%     newdims(newdims == dims0) = 1;
%     xprime_s = repmat(xprime_s, newdims);
% 
%     %% ----------------------------------------------------
%     % CONSTRUCT EXPECTATIONS MATRIX, ETC...
%     % -----------------------------------------------------
%     Emat = kron(income.ytrans_live, speye(p.nx));
% 
%     % Initial guess for consumption function
%     tempt_adj = 0.5 * (max(p.temptation) > 0.05);
%     con = (max(heterogeneity.r_broadcast, 0.001) + tempt_adj) .* xmat;
%     con = con(:);
%     con(con<=0) = min(con(con>0));
%     con = reshape(con, ss_dims);
% 
%     %% ----------------------------------------------------
%     % EGP ITERATION
%     % ----------------------------------------------------- 
%     iter = 1;
%     cdiff = 1;
%     conlast = con;
%     while iter<p.max_iter && cdiff>p.tol_iter
%         if iter > 1
%             conlast = conupdate;
%         end
%         iter = iter + 1;
%         c_xp = get_marginal_util_cons(...p, income, grids, c_xp)
%     end
% end
% end
% 
% 
% 
% 
% % 
% % function model = solve_EGP(p, grids, heterogeneity, income, futureshock, periods_until_shock, nextmodel)
% %     Same structure as before with if-else logic for different consumption cases
% % 
% %     % Setup initial useful objects and arrays
% %     nextmpcshock = (periods_until_shock == 1) * futureshock;
% %     ss_dims = [p.nx, p.nyP, p.nyF, p.nz];
% %       ss_dims = [p.nx, p.nyP, p.nyF, p.nz];
% %     R_bc = heterogeneity.R_broadcast;
% %     phi = heterogeneity.phi_broadcast;
% %     If expected future shock is negative, need to raise today's
% %     borrowing limit to satisfy future budget constraints with prob one
% %     (in all other cases, it should hold that adj_borr_lims = p.borrow_lim).
% %     tmp = p.borrow_lim - futureshock - income.minnety;
% %     adj_borr_lims = max(tmp ./ reshape(p.R, 1, []), p.borrow_lim);
% % 
% %     adj_borr_lims_bc = adj_borr_lims(:);
% %     if numel(adj_borr_lims_bc) == 1
% %         adj_borr_lims_bc = repmat(adj_borr_lims_bc, p.nz, 1);
% %     end
% %     adj_borr_lims_bc = shiftdim(adj_borr_lims_bc, -3);
% % 
% %     svecs_bc = grids.s.vec + (adj_borr_lims_bc - p.borrow_lim);
% %     svecs_bc_tax = p.compute_savtax(svecs_bc);
% % 
% %     xmat = grids.x.matrix + R_bc .* (adj_borr_lims_bc - p.borrow_lim);
% % 
% %     tempt_expr = heterogeneity.temptation_broadcast;
% %     tempt_expr = tempt_expr ./ (1 + tempt_expr);
% % 
% %     xprime as a function of s
% %     xprime_s = R_bc .* svecs_bc + income.netymatEGP + nextmpcshock;
% % 
% %     Extend to full state space
% %     newdims = [ss_dims p.nyT];
% %     [n1, n2, n3, n4, n5] = size(xprime_s);
% %     dims0 = [n1, n2, n3, n4, n5];
% %     newdims(newdims == dims0) = 1;
% %     xprime_s = repmat(xprime_s, newdims);
% % 
% %     % ----------------------------------------------------
% %     CONSTRUCT EXPECTATIONS MATRIX, ETC...
% %     -----------------------------------------------------
% %     Emat = kron(income.ytrans_live, speye(p.nx));
% % 
% %     Initial guess for consumption function
% %     tempt_adj = 0.5 * (max(p.temptation) > 0.05);
% % 
% %     con = (max(heterogeneity.r_broadcast, 0.001) + tempt_adj) .* xmat;
% %     con = con(:);
% %     con(con<=0) = min(con(con>0));
% %     con = reshape(con, ss_dims);
% % 
% %     % EGP Iteration
% %     iter = 1;
% %     cdiff = 1;
% %     conlast = con;
% % 
% %     while iter < p.max_iter && cdiff > p.tol_iter
% %         if iter > 1
% %             conlast = conupdate;
% %         end
% %         iter = iter + 1;
% % 
% %         Interpolate to get c_xp based on phi
% %         if p.phi == 0
% %             One good case
% %             c_xp = get_c_xprime_single_good(p, grids, xprime_s, nextmodel, conlast, nextmpcshock, xmat);
% %         else
% %             Two goods case
% %             [c1_xp, c2_xp] = get_c_xprime_two_goods(p, grids, xprime_s, nextmodel, conlast, nextmpcshock, xmat);
% %         end
% % 
% %         Marginal utility calculation based on phi
% %         if p.phi == 0
% %             muc_s = get_marginal_util_single_good(p, income, grids, c_xp, xprime_s, R_bc, Emat, heterogeneity, tempt_expr, svecs_bc);
% %         else
% %             muc_s = get_marginal_util_two_goods(p, income, grids, c1_xp, c2_xp, xprime_s, R_bc, Emat, heterogeneity, tempt_expr, svecs_bc, p.phi);
% %         end
% % 
% %         Consumption policy update based on marginal utility
% %         con_s = aux.u1inv(heterogeneity.risk_aver_broadcast, muc_s);
% % 
% %         Update saving policy based on phi
% %         if p.phi == 0
% %             sav = get_saving_policy_single_good(p, grids, con_s, nextmpcshock, R_bc, svecs_bc, xmat);
% %         else
% %             sav = get_saving_policy_two_goods(p, grids, con_s, nextmpcshock, R_bc, svecs_bc, xmat);
% %         end
% % 
% %         sav_tax = p.compute_savtax(sav);
% %         conupdate = xmat - sav - sav_tax;
% % 
% %         cdiff = max(abs(conupdate(:) - conlast(:)));
% %         if mod(iter, 100) == 0 && ~p.calibrating
% %             disp(['EGP Iteration ' int2str(iter) ', max con fn diff: ' num2str(cdiff)]);
% %         end
% %     end
% % 
% %     if cdiff > p.tol_iter
% %         model.EGP_cdiff = cdiff;
% %         return;
% %     end
% % 
% %     model.sav = sav;
% %     model.con = conupdate;
% %     model.EGP_cdiff = cdiff;
% % 
% %     Create interpolants for the optimal policy functions
% %     create_interpolants(model, p, grids, xmat, sav, conupdate, adj_borr_lims_bc);
% % 
% % end
% % function c_xp = get_c_xprime_single_good(p, grids, xp_s, nextmodel, conlast, nextmpcshock, xmat)
% %     Single good: interpolation logic remains the same
% %     c_xp = zeros(size(xp_s));
% %     dims_nx_nyT = [p.nx, 1, 1, 1, p.nyT];
% % 
% %     for ib = 1:p.nz
% %         for iyF = 1:p.nyF
% %             for iyP = 1:p.nyP
% %                 xp_s_ib_iyF_iyP = xp_s(:, iyP, iyF, ib, :);
% %                 if isempty(nextmodel)
% %                     coninterp = griddedInterpolant(xmat(:, iyP, iyF, ib), conlast(:, iyP, iyF, ib), 'linear');
% %                     c_xp(:, iyP, iyF, ib, :) = reshape(coninterp(xp_s_ib_iyF_iyP(:)), dims_nx_nyT);
% %                 else
% %                     c_xp(:, iyP, iyF, ib, :) = reshape(nextmodel.coninterp_ext{iyP, iyF, ib}(xp_s_ib_iyF_iyP(:)), dims_nx_nyT);
% %                 end
% %             end
% %         end
% %     end
% % end
% % 
% % function [c1_xp, c2_xp] = get_c_xprime_two_goods(p, grids, xp_s, nextmodel, conlast, nextmpcshock, xmat)
% %     Two goods: interpolate both c_1 and c_2
% %     c1_xp = zeros(size(xp_s));
% %     c2_xp = zeros(size(xp_s));
% %     dims_nx_nyT = [p.nx, 1, 1, 1, p.nyT];
% % 
% %     for ib = 1:p.nz
% %         for iyF = 1:p.nyF
% %             for iyP = 1:p.nyP
% %                 xp_s_ib_iyF_iyP = xp_s(:, iyP, iyF, ib, :);
% %                 if isempty(nextmodel)
% %                     coninterp_c1 = griddedInterpolant(xmat(:, iyP, iyF, ib), conlast(:, iyP, iyF, ib), 'linear');
% %                     coninterp_c2 = griddedInterpolant(xmat(:, iyP, iyF, ib), conlast(:, iyP, iyF, ib), 'linear');
% %                     c1_xp(:, iyP, iyF, ib, :) = reshape(coninterp_c1(xp_s_ib_iyF_iyP(:)), dims_nx_nyT);
% %                     c2_xp(:, iyP, iyF, ib, :) = reshape(coninterp_c2(xp_s_ib_iyF_iyP(:)), dims_nx_nyT);
% %                 else
% %                     c1_xp(:, iyP, iyF, ib, :) = reshape(nextmodel.coninterp_ext_c1{iyP, iyF, ib}(xp_s_ib_iyF_iyP(:)), dims_nx_nyT);
% %                     c2_xp(:, iyP, iyF, ib, :) = reshape(nextmodel.coninterp_ext_c2{iyP, iyF, ib}(xp_s_ib_iyF_iyP(:)), dims_nx_nyT);
% %                 end
% %             end
% %         end
% %     end
% % end
% % 
% % function muc_s = get_marginal_util_single_good(...
% % 	p, income, grids, c_xp, xp_s, R_bc,...
% %     Emat, betagrid_bc, risk_aver_bc, tempt_expr,...
% %     svecs_bc)
% % 
% %     savtaxrate = (1 + p.savtax .* (svecs_bc >= p.savtaxthresh));
% % 
% % 	First get marginal utility of consumption next period
% % 	muc_c = aux.utility1(risk_aver_bc, c_xp);
% % 
% %     muc_tempt = -tempt_expr .* aux.utility1(risk_aver_bc, xp_s+1e-7);
% %     mucnext = reshape(muc_c(:) + muc_tempt(:), [], p.nyT);
% % 
% %     Integrate
% %     expectation = Emat * mucnext * income.yTdist;
% %     expectation = reshape(expectation, [p.nx, p.nyP, p.nyF, p.nz]);
% % 
% %     muc_c_today = R_bc .* betagrid_bc .* expectation;
% %     muc_beq = aux.utility_bequests1(p.bequest_curv, p.bequest_weight,...
% %         p.bequest_luxury, svecs_bc);
% % 
% %     muc_s = (1 - p.dieprob) * muc_c_today ./ savtaxrate ...
% %         + p.dieprob * muc_beq;
% % end
% % 
% % 
% % function muc_s = get_marginal_util_two_goods(p, income, grids, c1_xp, c2_xp, xp_s, R_bc, Emat, heterogeneity, tempt_expr, svecs_bc, phi)
% %     Two goods: marginal utility with u(c_1) + \phi u(c_2)
% %     savtaxrate = (1 + p.savtax .* (svecs_bc >= p.savtaxthresh));
% % 
% %     Marginal utility for each good
% %     muc_c1 = aux.utility1(heterogeneity.risk_aver_broadcast, c1_xp);
% %     muc_c2 = aux.utility1(heterogeneity.risk_aver_broadcast, c2_xp);
% %     muc_tempt = -tempt_expr .* aux.utility1(heterogeneity.risk_aver_broadcast, xp_s + 1e-7);
% %     assignin('base', 'muc_c1', muc_c1);
% %     assignin('base', 'muc_c2', muc_c2);
% %     assignin('base', 'phi', phi);
% %     assignin('base','muc_tempt', muc_tempt);
% %     assignin('base', 'p', p.nyT);
% %     mucnext = reshape(muc_c1(:) + phi .* muc_c2(:) + muc_tempt(:), [], p.nyT);
% %     assignin('base', 'Emat', Emat);
% %     assignin('base', 'mucnext', mucnext);
% %     assignin('base', 'yTdist', income.yTdist);
% %     foo()
% %     expectation = Emat * mucnext * income.yTdist;
% %     expectation = reshape(expectation, [p.nx, p.nyP, p.nyF, p.nz]);
% % 
% %     muc_s = R_bc .* heterogeneity.betagrid_broadcast .* expectation ./ savtaxrate;
% % end
% % 
% % function sav = get_saving_policy_single_good(p, grids, x_s, nextmpcshock, R_bc,...
% %     svecs_bc, xmat)
% % 	Finds s(x), the saving policy function on the
% % 	cash-on-hand grid
% % 
% %     sav = zeros(size(x_s));
% %     xstar = zeros(p.nyP,p.nyF,p.nz);
% %     for ib  = 1:p.nz
% %     for iyF = 1:p.nyF
% %     for iyP = 1:p.nyP
% %         adj = xmat(:,iyP,iyF,ib) < x_s(1,iyP,iyF,ib);
% %         sav(adj,iyP,iyF,ib) = svecs_bc(1,1,1,ib);
% % 
% %         savinterp = griddedInterpolant(x_s(:,iyP,iyF,ib),...
% %             svecs_bc(:,1,1,ib), 'linear');
% % 
% %         sav(~adj,iyP,iyF,ib) = savinterp(...
% %             xmat(~adj,iyP,iyF,ib));
% %     end
% %     end
% %     end
% % end
% % 
% % function sav = get_saving_policy_two_goods(p, grids, con_s, nextmpcshock, R_bc, svecs_bc, xmat)
% %     Finds s(x), the saving policy function on the cash-on-hand grid
% %     when there are two consumption goods (c_1 and c_2)
% % 
% %     sav = zeros(size(con_s)); % Initialize the saving array
% %     xstar = zeros(p.nyP, p.nyF, p.nz); % Initialize
% % 
% %     for ib = 1:p.nz
% %         for iyF = 1:p.nyF
% %             for iyP = 1:p.nyP
% %                 Handle both c_1 and c_2
% %                 x_s_ib_iyF_iyP = svecs_bc(:, 1, 1, ib) + svecs_bc(:, 1, 1, ib);
% %                 adj = xmat(:, iyP, iyF, ib) < x_s_ib_iyF_iyP;
% % 
% %                 If consumption is below the state, set saving to the first point on the grid
% %                 sav(adj, iyP, iyF, ib) = svecs_bc(1, 1, 1, ib);
% % 
% %                 Interpolant for savings
% %                 savinterp = griddedInterpolant(x_s_ib_iyF_iyP, svecs_bc(:, 1, 1, ib), 'linear');
% % 
% %                 Apply saving policy for states that are above the adjustment threshold
% %                 sav(~adj, iyP, iyF, ib) = savinterp(xmat(~adj, iyP, iyF, ib));
% %             end
% %         end
% %     end
% % end
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
