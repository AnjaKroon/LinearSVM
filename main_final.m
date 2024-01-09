clear all;

%% DESCRIPTION
% Solving Linear SVM with three approaches

%% README
% Inputs: linear_svm.mat
% X_test            900x2             14400  double
% X_train           100x2              1600  double
% labels_test       900x1              7200  double
% labels_train      100x1               800  double

% Outputs:
% See terminal window for results of three solving approaches

%% IMPORT DATA
load("linear_svm.mat", "labels_train", "labels_test", "X_test", "X_train")
whos

%% HARD SVM CVX
disp(['-------- Solving with CVX solver --------']);
% Solve and plot on training set
% [w_cvx, b_cvx] = solve_cvx(X_train, labels_train);
%plotting(X_train, labels_train, w_cvx, b_cvx, 'CVX train: ');
%disp(['-------- Results CVX train --------']);
%result_cvx_train = eval(X_train, labels_train, w_cvx, b_cvx);
%disp("Objective Function Value: " + num2str(calc_opt(w_cvx)));

% Solve and plot on test set
[w_cvx_test, b_cvx_test] = solve_cvx(X_test, labels_test);
plotting(X_test, labels_test, w_cvx_test, b_cvx_test, 'CVX test: ');
disp(['-------- Results CVX test --------']);
result_cvx_test = eval(X_test, labels_test, w_cvx_test, b_cvx_test);
disp("Objective Function Value: " + num2str(calc_opt(w_cvx_test)));


%% Soft SVM Gradient Descent
disp(['-------- Solving Soft SVM with GD --------']);
% Initialize hyperparams
C = 1;
lr = 0.001;
max_num_iter = 10000;
epsilon = 0.001;

% Solve and plot on training set
%[w_grad, b_grad] = gradDesc(X_train, labels_train, C, lr, max_num_iter, epsilon);
%plotting(X_train, labels_train, w_grad,b_grad, 'GD train: ');
%disp(['-------- Results GD train --------']);
%result_grad_train = eval(X_train, labels_train, w_grad, b_grad);
%disp("Objective Function Value: " + num2str(calc_opt(w_grad)));

% Solve and plot on test set
[w_grad_test, b_grad_test] = gradDesc(X_train, labels_train, C, lr, max_num_iter, epsilon);
plotting(X_test, labels_test, w_grad_test,b_grad_test, 'GD test: ');
% plot_wts_b(w_set,b_set);
disp(['-------- Results GD test --------']);
result_grad_test = eval(X_test, labels_test, w_grad_test, b_grad_test);
disp("Objective Function Value: " + num2str(calc_opt(w_grad_test)));


%% Hard SVM Newton Step with Log Barrier
disp(['-------- Solving with Log Barrier NT --------']);
% TRAIN SET
% Finding feasible point
%[w_feas,b_feas] = solve_feasible(X_train,labels_train);
%plotting(X_train, labels_train, w_feas,b_feas, 'nt train feasible point: ');

% Initialize hyperparams
t=0.075;
iter_t=9;
mu=0.97;

%[w_hsvm_nt, b_hsvm_nt, w_set_hsvm_nt, b_set_hsvm_nt, cost] = hard_svm_nt(X_train, labels_train, w_feas, b_feas, t, mu, iter_t);
% disp("Objective Function Value: " + num2str(calc_opt(w_hsvm_nt)));
%plotting(X_train, labels_train, w_feas, b_feas, 'feasible initial point: ');
%plotting(X_train, labels_train, w_hsvm_nt, b_hsvm_nt, 'NT: ');
%plot_wts_b(w_set_hsvm_nt,b_set_hsvm_nt);
%disp(['-------- Results NT train --------']);
%result_grad_test = eval(X_test, labels_test, w_hsvm_nt, b_hsvm_nt);
%disp("Objective Function Value: " + num2str(calc_opt(w_hsvm_nt)));


% TEST SET
% Finding feasible point
[w_feas_test, b_feas_test] = solve_feasible(X_test, labels_test);
plotting(X_test, labels_test, w_feas_test, b_feas_test, 'NT test feasible point: ');

% Initialize hyperparams
t=0.0099;
iter_t=25;
mu=0.97;

[w_hsvm_nt_test, b_hsvm_nt_test, w_set_hsvm_nt_test, b_set_hsvm_nt_test, cost_test] = hard_svm_nt(X_test, labels_test, w_feas_test, b_feas_test, t, mu, iter_t);
plotting(X_test, labels_test, w_hsvm_nt_test, b_hsvm_nt_test, 'NT: ');
disp(['-------- Results NT train --------']);
result_grad_test = eval(X_test, labels_test, w_hsvm_nt_test, b_hsvm_nt_test);
disp("Objective Function Value: " + num2str(calc_opt(w_hsvm_nt_test)));


%% ANALYSIS
%disp(['-------- TRAIN SET Analysis on Final Width --------']);
%disp(['HARD SVM']);
%svmdistances(X_train, labels_train, w_cvx, b_cvx,"-------- CVX train set distances:  --------")
% svmdistances(X_train,labels_train, w_feas, b_feas,"-------- NT train set initial point distances:  --------")
%svmdistances(X_train, labels_train, w_hsvm_nt, b_hsvm_nt,"-------- NT train set distances:  --------")
%disp(['SOFT SVM']);
%svmdistances(X_train, labels_train, w_grad, b_grad, "-------- GD train set distances:  --------")

disp(['-------- TEST SET Analysis on Final Width --------']);
disp(['HARD SVM']);
svmdistances(X_test, labels_test, w_cvx_test, b_cvx_test, "-------- CVX test set distances:  --------")
svmdistances(X_test, labels_test, w_feas_test, b_feas_test, "-------- NT initial distances:  --------")
svmdistances(X_test, labels_test, w_hsvm_nt_test, b_hsvm_nt_test, "-------- NT test set distances:  --------")
disp(['SOFT SVM']);
svmdistances(X_test, labels_test, w_grad_test, b_grad_test, "-------- GD test set distances:  --------")


%% SUBFUNCTIONS
function [w, b] = solve_cvx(X, labels)
    disp(['Solving with CVX solver']);
    
    % start function timing
    tStart = cputime;
    
    % initial params
    [r,c] = size(X);
    
    % starting CVX solver
    cvx_begin
    variables w(c) b;
    minimize (norm(w)/2)
    %minimize (square_pos(norm(w)))
    subject to
    labels .* (X * w + b) >= 1;
    cvx_end
    
    % end function timing
    tEnd = cputime - tStart;
    disp(['Total CPU time: ' num2str(tEnd)]);
end

%function to find a feasible start point for log barrier method
function [w, b] = solve_feasible(X, labels)
    disp(['Finding a feasible point with CVX solver']);
    
    % start function timing
    tStart = cputime;
    
    % initial params
    [r,c] = size(X);
    
    % starting CVX solver
    cvx_begin
    variables w(c) b s;
    minimize (1)
    % minimize (square_pos(norm(w)))
    subject to
    1- labels .* (X * w + b) <= 0;
    %s>=0;
    cvx_end
    
    % end function timing
    tEnd = cputime - tStart;
    disp(['Total CPU time: ' num2str(tEnd)]);
end

function [w, b] = gradDesc(X, labels, C, lr, max_num_iter, epsilon)
    disp(['Solving with gradient descent']);
    
    % start function timing
    tStart = cputime;
    
    % initial params
    [r,c] = size(X);
    w = zeros(c,1);
    b = 0;
    prev_loss = inf;
    
    for iteration = 1:max_num_iter
        loss = 0;
        for i = 1:r
            % checking if current sample classified wrong
            if (labels(i) * (X(i,:) * w + b)) < 1
                % updating sub gradients using prev values
                dw = (w - C * labels(i) * X(i,:)');
                db = (- C * labels(i));
    
                % updating with projected subgrad
                w = (w - lr * dw);
                b = b - lr * db;
    
                % calculating total loss for current iteration
                % loss = loss + max(0, 1 - yi ( w * x + b)
                loss = loss + 1 - labels(i) * (X(i,:) * w + b);
    
            else  % loss function = 0, correctly classified
                dw = (w);
                % no need to update db/b
                w = (w - lr * dw);
    
            end
    end

    % Early stopping based on loss changes from prev. iteration
    if abs(prev_loss - loss) < epsilon
        disp(['Early stopping occuring at iteration: ' num2str(iteration)]);
        break;
    end

    % for early stopping criteria
    prev_loss = loss;
    end

    % end function timing and display results
    tEnd = cputime - tStart;
    disp(['Total CPU time: ' num2str(tEnd)]);
    disp(['Max iterations obtained unless early stopping: ' num2str(max_num_iter)]);
end

%log barrier with the newton step
function [w,b,w_array,b_array,cost]= hard_svm_nt(X,labels,w_feas,b_feas,t,mu,iter_t)
    disp(['Solving hard SVM with Newton Step']);
    tStart= cputime;
    
    % Initialize arrays
    w_array=zeros(2,1);
    b_array=zeros(1,1);
    
    % Set to incoming feasible values calculated by CVX
    w=w_feas;
    b=b_feas;
    cost=0;
    
    for iter= 1:iter_t
        if iter==1
            num_iter2=15;
        else
            num_iter2=4;
        end
        for iter2=1:num_iter2
            log_sum=0;
            sum1=0;
            sum2=0;
            sum3=0;
            sum4=0;
            for i=1:length(X)
                sum1=   sum1    +   (labels(i)*X(i,:)')./(labels(i)*(dot(w,X(i,:))+b)-1);
                sum2=   sum2    +   (labels(i))./(labels(i)*(dot(w,X(i,:))+b)-1);
                sum3=   sum3    +   (labels(i)*X(i,:)')*(labels(i)*X(i,:)')'  ./ ((1-labels(i)*(dot(w,X(i,:))+b)).^2);
                sum4=   sum4    +   labels(i).^2 ./ ((1-labels(i)*(dot(w,X(i,:))+b)).^2);
    
                log_sum = log_sum  + log((labels(i)*(dot(w,X(i,:))+b)-1));
            end
            
            % Updating the gradients
            dw=  t*w -  sum1;
            db=   -sum2;
            
            % Updating the second deriv
            d2w= t.*eye(2)  + sum3;
            d2b= sum4;
            
            % Calculating Newton step values
            wnt = -d2w\dw;
            bnt = -d2b\db;
            
            % Updating Newton step values
            w=  w   +    wnt;
            b=  b   +    bnt;
            
            % Calculate cost value and store the value of w and b into long
            % term tracking
            cost_iter= t*norm(w).^2 - log_sum;
            cost=vertcat(cost,cost_iter);
            w_array=horzcat(w_array,w);
            b_array=horzcat(b_array,b);
        end
        % Update t value
        t= mu*t;
    end
    tEnd = cputime - tStart;
    disp(['Total CPU time: ' num2str(tEnd)]);
end


function [w, b] = Grad(X, labels, C, lr, max_num_iter, epsilon)
    % disp(['Solving with Grad']);
    
    % start function timing
    tStart = cputime;
    
    % Initialize function values
    [r,c] = size(X);
    w = zeros(c,1);
    b = 0;
    prev_loss = inf;
    
    for iteration = 1:max_num_iter
        loss = 0;
        dw = zeros(c,1);
        db = 0;
    
        for i = 1:r
            % checking if current sample classified wrong
            if (labels(i) * (X(i,:) * w + b)) < 1
                dw = (w - C * labels(i) * X(i,:)');
                db = (- C * labels(i));
    
                w = w - lr * dw;
                b = b - lr * db;
                % updating sub gradients using prev values
                %dw = (w - C * labels(i) * X(i,:)');
                %db = (- C * labels(i));
                % add to loss
                loss = loss + 1 - labels(i) * (X(i,:) * w + b);
    
            end
    
        end
        % no update of sub gradients when point classified correctly
        % updating w and b
        w = w - lr * dw;
        b = b - lr * db;
    
        % Early stopping based on loss changes from prev. iteration
        if abs(prev_loss - loss) < epsilon
            disp(['Early stopping occuring at iteration: ' num2str(iteration)]);
            break;
        end
        prev_loss = loss;
    end
    
    tEnd = cputime - tStart;
    disp(['Total CPU time: ' num2str(tEnd)]);
    disp(['Max iterations obtained unless early stopping: ' num2str(max_num_iter)]);
end

function plotting(X, labels, w, b, algo_name)
    % Getting class groupings, color coding points
    all = horzcat(X, labels);
    red = [];
    green = [];
    for i = 1:length(X)
        if all(i, 3) == 1
            red = [red; all(i, :)];
        else
            green = [green; all(i, :)];
    end
    end

    % Plot the corresponding line
    x_values = linspace(floor(min(X(:, 1))), round(max(X(:, 1))), 1000);
    y_values = ( - w(2) * x_values - b ) / w(1);
    
    figure;
    plot(y_values, x_values, 'b', 'LineWidth', 2);
    hold on;
    scatter(red(:, 1), red(:, 2), 'o', 'r');
    hold on;
    scatter(green(:, 1), green(:, 2), 'o', 'g');
    curDataName = inputname(1);
    title(sprintf('%s %s Data Points and Optimal Separational Hyperplane (SVM)', algo_name, curDataName));
end

%plot the norm and bias of the weight vectors, saved at each iteration.
function plot_wts_b(w_s,b_s)
    figure(3);
    plot(vecnorm(w_s(:,2:end)));
    title("Weight Norms")
    figure(4);
    plot(b_s);
    title("Biases")
end

function [correctly_classified] = eval(X, labels, w, b)
    % Determining the "calculated classifications"
    dec_boundary = X * w + b;
    classifs = sign(dec_boundary);
    
    % Computing metric to compare
    len = size(classifs(:,1));
    count = 0;
    for i = 1:len
        % Comparing the calculated classifications to the true labels
        if (classifs(i) == labels(i))
            count = count + 1;
        end
    end
    correctly_classified = count/len(:,1);
    disp(['Of ' num2str(len(:,1)) ', ' num2str(count) ' points were correctly classified.']);
    disp(['Correctly classified? [0, 1]: ' num2str(correctly_classified)]);
end

%calculates the objective function value
function opt_point =calc_opt(w)
    opt_point = norm(w);
end

%calculates the distance between the support vectors and hyperplane
function distance = svmdistances(data,labels,w,b,alg)
    d1 = zeros(1);
    dm1 = zeros(1);
    w_norm = w./norm(w);
    b_norm = b./norm(w);
    for i = 1: length(data)
        yi = labels(i);
        if(yi == 1)
            dist1 = yi.*(dot(w_norm,data(i,:))+b_norm);
            d1 = vertcat(d1,dist1);
        else
            distm1 = yi.*(dot(w_norm,data(i,:))+b_norm);
            dm1 = vertcat(dm1,distm1);
        end
    end
    min_d1 = min(d1(2:end));
    min_dm1 = min(dm1(2:end));
    distance = min_d1+min_dm1;
    disp(alg)
    disp("distance to +ve class " + min_d1)
    disp("distance to -ve class " + min_dm1)
    disp("distance between classes " + distance)
end