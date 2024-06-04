function [ wA, bA, EXITFLAG] = LTWSVM1( xA, xB, C1 )
%LTWSVM1 Solves the Linear Twin SVM Dual QPP for the first plane

[N1,D]=size(xA);
[N2,D]=size(xB);


H=[xA,ones(N1,1)];
G=[xB,ones(N2,1)];

alpha0=[rand(N2,1)];

% Quadratic term objective
obj_quad=G*pinv(H'*H+eps*eye(size(H'*H)))*G';
obj_quad=obj_quad+eps*eye(size(obj_quad)); %Conditioning
obj_quad=(obj_quad+obj_quad')/2; %Making symmetric

% Linear term objective
obj_linear=-ones(size(alpha0,1),1);


% Setup inwquality constraints
A_ineq_const=[];
b_ineq_const=[];

% Setup equality constraints
A_eq_const=[];
b_eq_const=[];

% Setup bounds
lb=zeros(size(alpha0,1),1);
ub=C1*ones(size(alpha0,1),1);

try
    % Setup options
    options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','none');
    % [X, FVAL, EXITFLAG]=quadprog(obj_quad, obj_linear, A_ineq_const, b_ineq_const, A_eq_const, b_eq_const, lb, ub);

    % Solve QPP
    [X, FVAL, EXITFLAG]=quadprog(obj_quad, obj_linear, A_ineq_const, b_ineq_const, A_eq_const, b_eq_const, lb, ub, [], options);
    % EXITFLAG =1;
    % Compute solution
    u=-pinv(H'*H + eps*eye(size(H'*H)))*G'*X;
    
    wA=u(1:end-1,:);
    bA=u(end,:);
    % wA=u(:,1:end-1);
    % bA=u(:,end);
    % disp(size(wA));
    % disp(size(bA));
catch
    wA=rand(N1+N2,1);
    bA=rand;
    % wA=rand(1,N1+N2);
    % bA=rand;
end
end

