function c = u2inv(risk_aver,u, phi)
	c = (u ./ (1 .+ phi)) .^ (1 ./ risk_aver);
end