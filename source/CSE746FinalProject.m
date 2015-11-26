fileID = fopen('problem.txt','r');
formatSpec = '%f';
Raw = fscanf(fileID,formatSpec);
fclose(fileID);


M = Raw(1);
N = Raw(2);

f = Raw(3:3+N-1);
beq = Raw(3+N:3+N+M-1);
A = Raw(3+N+M:3+N+M+M*N-1);
Aeq = reshape(A,N,M)';

lb = zeros(size(f));
tic
options = optimoptions('linprog','Algorithm','simplex');
[x,fval,exitflag,output,lambda] = linprog(-f,[],[],Aeq,beq,lb,[],[],options)
toc


