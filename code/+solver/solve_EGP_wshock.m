function model = solve_EGP_wshock(p, grids, heterogeneity,...
    income, futureshock, periods_until_shock, nextmodel)

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
    con1 = (max(heterogeneity.r_broadcast, 0.001) + tempt_adj) .* xmat...
        ./ (1 + p.phi);
    con1 = con1(:);
    con1(con1<=0) = min(con1(con1>0));
    con1 = reshape(con1, ss_dims);

    con2 = p.phi .* (max(heterogeneity.r_broadcast, 0.001) + tempt_adj)...
        .* xmat ./ (1 + p.phi);
    con2 = con2(:);
    pos_vals = con2(con2>0);
    if ~isempty(pos_vals)
        con2(con2<=0) = min(pos_vals);
    end
    con2 = reshape(con2, ss_dims);

    iter = 1;
    cdiff = 1;
    conlast1 = con1;
    conlast2 = con2;
    con = conlast1 + conlast2;
    conlast = con;
    
    cdiff_history1 = zeros(p.max_iter, 1);
    cdiff_history2 = zeros(p.max_iter, 1);

    phi = 0;
    while iter < p.max_iter && cdiff>p.tol_iter
        if iter > 1
            conlast = conupdate;
            conlast1 = conupdate1;
            conlast2 = conupdate2;
        end
        iter = iter + 1;

        phi = new_phi(p);
        
        %Interpolating to get c_1(x') using c(x)
        c_xp1 = get_c_xprime(p, grids, xprime_s,...
            nextmodel, conlast1, nextmpcshock, xmat);
        c_xp2 = get_c_xprime(p, grids, xprime_s,...
            nextmodel, conlast2, nextmpcshock, xmat);
        % assignin('base', 'c_xp1', c_xp1);
        % assignin('base', 'c_xp2', c_xp2);

        %Marginal utility of consumption in current period, from the Euler
        %equation
        muc_s1 = get_marginal_util_cons(...
            p, income, grids, c_xp1, xprime_s, R_bc,...
            Emat, heterogeneity.betagrid_broadcast,...
            heterogeneity.risk_aver_broadcast,...
            tempt_expr, svecs_bc);

        muc_s2 = get_marginal_util_cons2(...
            p, phi, income, grids, c_xp2, xprime_s, R_bc,...
            Emat, heterogeneity.betagrid_broadcast,...
            heterogeneity.risk_aver_broadcast,...
            tempt_expr, svecs_bc);
        % assignin('base', 'muc_s2', muc_s2);
        

        %%Get consumption of the two goods
        con_s1 = aux.u1inv(heterogeneity.risk_aver_broadcast, muc_s1)...
            ./ (1 + p.phi);
        con_s2 = p.phi .* con_s1;
        con_s2(isnan(con_s2)) = 0;

        % x(s) = s + stax + c1 + c2
        x_s = svecs_bc + svecs_bc_tax + con_s1 + con_s2;
        

        %interpolate from x(s) to get s(x)
        sav = get_saving_policy(p, grids, x_s, nextmpcshock,...
            R_bc, svecs_bc, xmat);
        sav_tax = p.compute_savtax(sav);

        %Update consumption fns
        conupdate = xmat - sav - sav_tax;
        conupdate1 = conupdate ./ (1 + p.phi);
        conupdate2 = conupdate1 .* p.phi;
       
  

        %%Update policies for both goods
        cdiff1 = max(abs(conupdate1(:) - conlast1(:)));
        cdiff2 = max(abs(conupdate2(:) - conlast2(:)));
        cdiff = max(cdiff1, cdiff2);
        %cdiff = max(abs(conupdate(:) - conlast(:)));

        cdiff_history1(iter) = cdiff1;
        cdiff_history2(iter) = cdiff2;
        p.phi = phi;

        if (mod(iter, 100) == 0) && ~p.calibrating
            disp(['  EGP Iteration ' int2str(iter), ' max con fn diff is ' num2str(cdiff)]);
            disp(['  cdiff1: ' num2str(cdiff1) ', cdiff2: ' num2str(cdiff2)]);
             if iter > 100
        last_100_diff1 = cdiff_history1(iter-100:iter);
        last_100_diff2 = cdiff_history2(iter-100:iter);
        disp(['  Average change in last 100 iterations:']);
        disp(['    cdiff1: ' num2str(mean(diff(last_100_diff1)))]);
        disp(['    cdiff2: ' num2str(mean(diff(last_100_diff2)))]);
    end
        end
    end
        
        if cdiff > p.tol_iter
            %EGP did not conv, don't find stationary dist.
            AYdiff =  10000;
            model.EGP_cdiff = cdiff;
            fprintf("Uh oh: %g", cdiff)
            return
        end    

        model.sav = sav;
        model.con1 = conupdate1;
        model.con2 = conupdate2;
        model.con = conupdate;
        model.EGP_cdiff = cdiff;

        %create interpolants from optimal policy functions and finding
        %saving values associated with xvals
        model.savinterp = cell(p.nyP,p.nyF,p.nz);
        model.coninterp = cell(p.nyP,p.nyF,p.nz);
        model.coninterp_ext = cell(p.nyP,p.nyF,p.nz);
        model.coninterp_mpc = cell(p.nyP,p.nyF,p.nz);

        model.coninterp1 = cell(p.nyP, p.nyF, p.nz);
        model.coninterp_ext1 = cell(p.nyP, p.nyF, p.nz);
        model.coninterp_mpc1 = cell(p.nyP, p.nyF, p.nz);

        model.coninterp2 = cell(p.nyP, p.nyF, p.nz);
        model.coninterp_ext2 = cell(p.nyP, p.nyF, p.nz);
        model.coninterp_mpc2 = cell(p.nyP, p.nyF, p.nz);
        for ib = 1:p.nz
        for iyF = 1:p.nyF
        for iyP = 1:p.nyP
            model.savinterp{iyP, iyF, ib} = griddedInterpolant(...
                xmat(:, iyP, iyF, ib), model.sav(:,iyP, iyF, ib),...
                'linear');
            model.coninterp1{iyP, iyF, ib} = griddedInterpolant(...
                xmat(:, iyP, iyF, ib), model.con1(:, iyP, iyF, ib),...
                'linear');
            model.coninterp2{iyP, iyF, ib} = griddedInterpolant(...
                xmat(:, iyP, iyF, ib), model.con2(:, iyP, iyF, ib),...
                'linear');
            model.coninterp{iyP, iyF, ib} = griddedInterpolant(...
                xmat(:, iyP, iyF, ib), model.con2(:, iyP, iyF, ib));

            cmin = model.con(1, iyP, iyF, ib);

            xmin = xmat(1, iyP, iyF, ib);
            cmin1 = model.con1(1, iyP, iyF, ib);
            cmin2 = model.con2(1, iyP, iyF, ib);
            lbound = 1e-7;

            model.coninterp_ext1{iyP, iyF, ib} = @(x) extend_interp(...
                model.coninterp1{iyP, iyF, ib}, x, xmin, cmin1, lbound,...
                adj_borr_lims_bc(1, 1, 1, ib));
            model.coninterp_ext2{iyP, iyF, ib} = @(x) extend_interp(...
                model.coninterp2{iyP, iyF, ib}, x, xmin, cmin2, lbound,...
                adj_borr_lims_bc(1, 1, 1, ib));
            model.coninterp{iyP, iyF, ib} = @(x) extend_interp(...
                adj_borr_lims_bc(1, 1, 1, ib));

            model.coninterp_mpc1{iyP, iyF, ib} = @(x) extend_interp(...
                model.coninterp1{iyP, iyF, ib}, x, xmin, cmin1, -inf,...
                adj_borr_lims_bc(1,1,1,ib));
            model.coninterp_mpc2{iyP, iyF, ib} = @(x) extend_interp(...
                model.coninterp2{iyP, iyF, ib}, x, xmin, cmin2, -inf,...
                adj_borr_lims_bc(1,1,1,ib));
            model.coninterp_mpc{iyP, iyF, ib} = @(x) extend_interp(...
                model.coninterp1{iyP, iyF, ib}, x, xmin, cmin, -inf,...
                adj_borr_lims_bc(1, 1, 1, ib));
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


function muc_s2 = get_marginal_util_cons2(...
	p, phi, income, grids, c_xp, xp_s, R_bc,...
    Emat, betagrid_bc, risk_aver_bc, tempt_expr,...
    svecs_bc)

    savtaxrate = (1 + p.savtax .* (svecs_bc >= p.savtaxthresh));
 
	% First get marginal utility of consumption next period
    %Uses phi here for the second good

	muc_c2 = phi .* aux.utility1(risk_aver_bc, c_xp);
    muc_tempt = -tempt_expr .* aux.utility1(risk_aver_bc, xp_s+1e-7);
    mucnext2 = reshape(muc_c2(:) + muc_tempt(:), [], p.nyT);

    % Integrate
    expectation = Emat * mucnext2 * income.yTdist;
    expectation = reshape(expectation, [p.nx, p.nyP, p.nyF, p.nz]);

    muc_c_today = R_bc .* betagrid_bc .* expectation;
    muc_beq = aux.utility_bequests1(p.bequest_curv, p.bequest_weight,...
        p.bequest_luxury, svecs_bc);

    muc_s2 = (1 - p.dieprob) * muc_c_today ./ savtaxrate ...
        + p.dieprob * muc_beq;
    muc_s2(isnan(muc_s2)) = 0;
    muc_s2(isinf(muc_s2)) = 0;
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
        try
        adj = xmat(:,iyP,iyF,ib) < x_s(1,iyP,iyF,ib);
        sav(adj,iyP,iyF,ib) = svecs_bc(1,1,1,ib);

        x_points = x_s(:, iyP, iyF, ib);
        v_points = svecs_bc(:, 1, 1, ib);
        if ~issorted(x_points)
            error('x_points must be monotonic')
        end

        savinterp = griddedInterpolant(x_s(:,iyP,iyF,ib),...
            svecs_bc(:,1,1,ib), 'linear');

        sav(~adj,iyP,iyF,ib) = savinterp(...
            xmat(~adj,iyP,iyF,ib));
        catch ME
            fprintf('Error at ib=%d, iyF=%d, iyP=%d: %s\n',...
                ib, iyF, iyP, ME.message);
            rethrow(ME)
        end
    end
    end
    end
end

%%gets expected phi value
function expected_phi = get_expected_phi(p)
    phil = p.phiL;
    phih = p.phiH;
    trans = p.phi_trans;
    expected_phi = p.phi;  % start with current values
    
    for i = 1:length(p.phi)
        if p.phi(i) == phil
            % When currently at phil, expected value is:
            expected_phi(i) = phil * trans(1,1) + phih * (1 - trans(1,1));
        else
            % When currently at phih, expected value is:
            expected_phi(i) = phil * (1 - trans(2,2)) + phih * trans(2,2);
        end
    end
end

%%check that when people are in the different level/state, check if their
%%consumption decision is changing.
%%calculates the new phi parameter, based on its transition matrix
function phi = new_phi(params)
    phi = params.phi;
    phil = params.phiL;
    phih = params.phiH;
    trans = params.phi_trans;
    for i = 1:length(params.phi)
         curr = phi(i);
         rand_num = rand;
         if curr == phil
             if rand_num > trans(1, 1)
                 phi(i) = phih;
             end
         else
             if rand_num < trans(2,2)
                 phi(i) = phil;
             end
         end
    end
end


