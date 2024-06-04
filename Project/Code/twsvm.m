function [w1, w2, b1, b2] = twsvm(X1, X2, c1, c2)
    % X1 and X2 are the data points of the two classes
    % c1 and c2 are the penalty parameters for the two problems

    % Number of samples in each class
    n1 = size(X1, 1);
    n2 = size(X2, 1);

    % Extended data matrices (add a column of ones for bias)
    X1 = [X1, ones(n1, 1)];
    X2 = [X2, ones(n2, 1)];

    % Construct the matrices for the quadratic programming problems
    H1 = [X1; c2 * X2];
    H2 = [X2; c1 * X1];

    % Construct identity matrices for the QP problems
    I1 = eye(n1 + n2);
    I2 = eye(n2 + n1);

    % Solve the first QP problem
    e1 = [zeros(n1, 1); ones(n2, 1)];
    A1 = H1' * H1 + I1;
    b1 = H1' * e1;
    w1 = pinv(A1) * b1;

    % Solve the second QP problem
    e2 = [zeros(n2, 1); ones(n1, 1)];
    A2 = H2' * H2 + I2;
    b2 = H2' * e2;
    w2 = pinv(A2) * b2;

    % Extract bias and weights
    b1 = w1(end);
    b2 = w2(end);
    w1 = w1(1:end-1);
    w2 = w2(1:end-1);

    % Display the model parameters
    fprintf('Model 1 (w1, b1): \n');
    disp(w1);
    disp(b1);
    fprintf('Model 2 (w2, b2): \n');
    disp(w2);
    disp(b2);
end