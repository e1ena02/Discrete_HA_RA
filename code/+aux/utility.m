function u = utility(risk_aver,c)
	if isequal(size(risk_aver),size(c))
		u = zeros(size(risk_aver));
		Ilog_u = (risk_aver == 1);

		u(Ilog_u) = log(c(Ilog_u));
		u(~Ilog_u) = (c(~Ilog_u) .^(1-risk_aver(~Ilog_u) )-1)./(1-risk_aver(~Ilog_u) );
	elseif numel(risk_aver) == 1
		if risk_aver == 1
			u = log(c);
		else
			u = (c .^(1-risk_aver) - 1) ./ (1 - risk_aver);
		end
    end
end

% function u = utility(risk_aver, c1, c2, phi)
%     % Computes utility with an expense shock
%     % risk_aver: risk aversion parameter
%     % c1: consumption before expense shock
%     % c2: consumption after expense shock
%     % phi: weight of post-shock consumption utility (typically small)
% 
%     % Initialize utility for u(c1) and u(c2)
%     u1 = zeros(size(c1));
%     u2 = zeros(size(c2));
% 
%     % Case when c1 and c2 have the same size as risk_aver
%     if isequal(size(risk_aver), size(c1))
%         Ilog_u = (risk_aver == 1);
% 
%         % Utility for u(c1)
%         u1(Ilog_u) = log(c1(Ilog_u));
%         u1(~Ilog_u) = (c1(~Ilog_u) .^ (1 - risk_aver(~Ilog_u)) - 1) ./ (1 - risk_aver(~Ilog_u));
% 
%         % Utility for u(c2) (post-shock)
%         u2(Ilog_u) = log(c2(Ilog_u));
%         u2(~Ilog_u) = (c2(~Ilog_u) .^ (1 - risk_aver(~Ilog_u)) - 1) ./ (1 - risk_aver(~Ilog_u));
%     elseif numel(risk_aver) == 1
%         % Case when risk_aver is scalar
%         if risk_aver == 1
%             u1 = log(c1);
%             u2 = log(c2);
%         else
%             u1 = (c1 .^ (1 - risk_aver) - 1) ./ (1 - risk_aver);
%             u2 = (c2 .^ (1 - risk_aver) - 1) ./ (1 - risk_aver);
%         end
%     end
% 
%     % Combined utility: u(c1) + phi * u(c2)
%     u = u1 + phi * u2;
% end
% 
% 
