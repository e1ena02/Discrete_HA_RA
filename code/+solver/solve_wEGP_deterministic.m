function norisk = solve_EGP_deterministic(p, grids,...
    heterogeneity, varargin)
    % This function uses the method of endogenous grid points to find the
    % policy functions of the deterministic model. Output is in the
    % 'norisk' structure.
    % Elena Elbarmi, 2024
    % elbarmi@uchicago.edu

    sgrid_bc = repmat(grids.s.vec, 1, p.nz);
    sgrid_tax = p.compute_savtax(sgrid_bc);
    msavtaxrate = (1 + p.savtax .* (sgrid_bc >= p.savtaxthresh));

    Emat = kron(heterogeneity.ztrans, speye(p.nx));
    r_bc = reshape(p.r, 1, []);
    R_bc = reshape(p.R, 1, []);
    phi = p.phiH;
    risk_aver_bc = reshape(p.risk_aver, 1, []);
    beta_bc = reshape(heterogeneity.betagrid, 1, []);

    tmp = p.temptation ./ (1 + p.temptation);
    tempt_bc = reshape(tmp, 1, []);

    % initial guess for consumption function
    tempt_adj = 0.5 * (max(p.temptation) > 0.05);
    r_bc_adj = max(r_bc, 0.001);
    con1 = ((r_bc_adj + tempt_adj) .* grids.x.matrix_norisk) ./ (1 + phi);
    con1 = con1(:);
    con1(con1<=0) = min(con1(con1>0));
    con1 = reshape(con1, [p.nx, p.nz]);

    con2 = phi .* ((r_bc_adj + tempt_adj) .* grids.x.matrix_norisk);
    con2 = con2(:);
    con2(con2<=0) = min(con2(con2>0));
    con2 = reshape(con2, [p.nx, p.nz]);

    con = ((r_bc_adj + tempt_adj) .* grids.x.matrix_norisk);
    con = con(:);
    con(con<=0) = min(con(con>0));
    con = reshape(con, [p.nx, p.nz]);

    iter = 0;
    cdiff = 1000;
    while (iter <= p.max_iter) && (cdiff > p.tol_iter)
        iter = iter + 1;
        
        conlast1 = con1;
        conlast2 = con2;
        conlast = con;

        muc1_next = aux.utility1(risk_aver, conlast1);
        muc2_next = phi .* aux.utility1(risk_aver, conlast2);

        % muc_next = aux.utility1(risk_aver_bc, conlast);
        % tempt_next = -tempt_bc .* aux.utility1(...
        %     risk_aver_bc, grids.x.matrix_norisk);
        % beq_next = aux.utility_bequests1(p.bequest_curv, p.bequest_weight,...
        %     p.bequest_luxury, sgrid_bc);

        expectation = Emat * (muc_next1(:));

        % expectation = Emat * (muc_next(:) - tempt_next(:));
        emuc_live = R_bc .* beta_bc .* reshape(expectation, [p.nx, p.nz]);

        muc_today = (1 - p.dieprob) * emuc_live ./ msavtaxrate ...
            + p.dieprob .* beq_next;

        con1_today = aux.u1inv(risk_aver_bc, muc_today) / (1 + phi);
        con2_today = phi .* con1_today;
        
        con_today = con1_today + con2_today;
        % muc_next = aux.utility1(risk_aver_bc, conlast);
        % tempt_next = -tempt_bc .* aux.utility1(...
        %     risk_aver_bc, grids.x.matrix_norisk);
        % beq_next = aux.utility_bequests1(p.bequest_curv, p.bequest_weight,...
        %     p.bequest_luxury, sgrid_bc);

        cash1 = con_today + sgrid_bc + sgrid_tax;
        
        sav = zeros(p.nx, p.nz);
        for ib = 1:p.nz
            savinterp = griddedInterpolant(cash1(:,ib), grids.s.vec, 'linear');
            sav(:,ib) = savinterp(grids.x.matrix_norisk(:,ib));

            adj = grids.x.matrix_norisk(:,ib) < cash1(1,ib);
            sav(adj,ib) = p.borrow_lim;
        end