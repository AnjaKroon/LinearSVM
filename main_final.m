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
[w_cvx, b_cvx] = solve_cvx(X_train, labels_train);
plotting(X_train, labels_train, w_cvx, b_cvx, 'CVX train: ');
disp(['-------- Results CVX train --------']);
result_cvx_train = eval(X_train, labels_train, w_cvx, b_cvx);
disp("Objective Function Value: " + num2str(calc_opt(w_cvx)));


%% Soft SVM Gradient Descent
disp(['-------- Solving Soft SVM with GD --------']);
% Initialize hyperparams
C = 1;
lr = 0.001;
max_num_iter = 10000;
epsilon = 0.001;

% Solve and plot on training set
[w_grad, b_grad] = gradDesc(X_train, labels_train, C, lr, max_num_iter, epsilon);
plotting(X_train, labels_train, w_grad,b_grad, 'GD train: ');
disp(['-------- Results GD train --------']);
result_grad_train = eval(X_train, labels_train, w_grad, b_grad);
disp("Objective Function Value: " + num2str(calc_opt(w_grad)));


%% Hard SVM Newton Step with Log Barrier
disp(['-------- Solving with Log Barrier NT --------']);
% TRAIN SET
% Finding feasible point
[w_feas,b_feas] = solve_feasible(X_train,labels_train);
plotting(X_train, labels_train, w_feas,b_feas, 'nt train feasible point: ');

%Initialize hyperparams
t=1;
iter_t=3000;
mu=1.001;
[w_hsvm_nt, b_hsvm_nt, w_set_hsvm_nt, b_set_hsvm_nt, cost] = hard_svm_nt(X_train, labels_train, w_feas, b_feas, t, mu, iter_t);
disp("Objective Function Value: " + num2str(calc_opt(w_hsvm_nt)));
plotting(X_train, labels_train, w_feas, b_feas, 'feasible initial point: ');
plotting(X_train, labels_train, w_hsvm_nt, b_hsvm_nt, 'NT: ');
plotting(X_test, labels_test, w_hsvm_nt, b_hsvm_nt, 'NT: ');
plot_wts_b(w_set_hsvm_nt,b_set_hsvm_nt);
disp(['-------- Results NT train --------']);
result_grad_test = eval(X_test, labels_test, w_hsvm_nt, b_hsvm_nt);
disp("Objective Function Value: " + num2str(calc_opt(w_hsvm_nt)));



%% ANALYSIS
disp(['-------- TRAIN SET Analysis on Final Width --------']);
disp(['HARD SVM']);
svmdistances(X_train, labels_train, w_cvx, b_cvx,"-------- CVX train set distances:  --------")
svmdistances(X_train,labels_train, w_feas, b_feas,"-------- NT train set initial point distances:  --------")
svmdistances(X_train, labels_train, w_hsvm_nt, b_hsvm_nt,"-------- NT train set distances:  --------")
disp(['SOFT SVM']);
svmdistances(X_train, labels_train, w_grad, b_grad, "-------- GD train set distances:  --------")


%% SUBFUNCTIONS
% solve_cvx: uses the CVX solver to find the optimal separating hyperplane
% returns weight and bias vector to describe separating hyperplane
function [w, b] = solve_cvx(X, labels)
    disp(['Solving with CVX solver']);
    
    tStart = cputime;
    
    [r,c] = size(X);
    
    cvx_begin
    variables w(c) b;
    minimize (norm(w)/2)
    %minimize (square_pos(norm(w)))
    subject to
    labels .* (X * w + b) >= 1;
    cvx_end
    
    tEnd = cputime - tStart;
    disp(['Total CPU time: ' num2str(tEnd)]);
end

% solve_feasible: finds a feasible start point for log barrier method
% returns weight and bias vector to describe separating hyperplane
function [w, b] = solve_feasible(X, labels)
    disp(['Finding a feasible point with CVX solver']);
    
    tStart = cputime;
    
    [r,c] = size(X);
    
    cvx_begin
    variables w(c) b s;
    minimize (1)
    % minimize (square_pos(norm(w)))
    subject to
    1- labels .* (X * w + b) <= 0;
    %s>=0;
    cvx_end
    
    tEnd = cputime - tStart;
    disp(['Total CPU time: ' num2str(tEnd)]);
end

% gradDesc: finds optimal separating hyperplane for the given data
% returns weight and bias vector to describe separating hyperplane
function [w, b] = gradDesc(X, labels, C, lr, max_num_iter, epsilon)
    disp(['Solving with gradient descent']);

    tStart = cputime;
    
    [r,c] = size(X);
    w = zeros(c,1);
    b = 0;
    prev_loss = inf;
    
    for iteration = 1:max_num_iter
        loss = 0;
        for i = 1:r
            % checking if current sample classified wrong
            if (labels(i) * (X(i,:) * w + b)) < 1
                % updating sub gradients 
                dw = (w - C * labels(i) * X(i,:)');
                db = (- C * labels(i));
    
                w = (w - lr * dw);
                b = b - lr * db;
    
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

    prev_loss = loss;
    end

    tEnd = cputime - tStart;
    disp(['Total CPU time: ' num2str(tEnd)]);
    disp(['Max iterations obtained unless early stopping: ' num2str(max_num_iter)]);
end

% hard_svm_nt: finds optimal separating hyperplane for the given data
% returns weight and bias vector to describe separating hyperplane
function [w,b,w_array,b_array,cost]= hard_svm_nt(X,labels,w_feas,b_feas,t,mu,iter_t)
    disp(['Solving hard SVM with Newton Step']);
    tStart= cputime;
    
    w_array=zeros(2,1);
    b_array=zeros(1,1);
    
    w=w_feas;
    b=b_feas;
    cost=0;
    for iter= 1:iter_t
    
        for iter2=1:30
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
    
        dw=  t*w -  sum1;
        db=   -sum2;
    
        d2w= t.*eye(2)  + sum3;
        d2b= sum4;
    
    
        wnt= -inv(d2w)*dw;
        bnt= -inv(d2b)*db;
    
        if(dw==0.0)
            disp(num2str("dw at 0")+dw);
            break;
        end
    
        w=  w   +  0.1*  wnt;
        b=  b   +  0.1*  bnt;
    
        cost_iter= t*norm(w).^2 - log_sum;
        cost=vertcat(cost,cost_iter);
    
        w_array=horzcat(w_array,w);
        b_array=horzcat(b_array,b);
    
        end
    t= mu*t;
    end
    
    
    tEnd = cputime - tStart;
    
    disp(['Total CPU time: ' num2str(tEnd)]);

end

% plotting: transforms given data into corresponding plot to visualize
% results of the optimization algorithms
% outputs plot popup window
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

% plot_wts_b: plot the norm and bias of the weight vectors, 
% saved at each iteration
% outputs plot popup window
function plot_wts_b(w_s,b_s)
    figure(3);
    plot(vecnorm(w_s(:,2:end)));
    title("Weight Norms")
    figure(4);
    plot(b_s);
    title("Biases")
end

% eval: evaluates the ability of the separating hyperplane to separate data
% displays number of points correctly classified in the terminal,
% returns a value between 0 and 1 where 1 indicates perfectly classified
function [correctly_classified] = eval(X, labels, w, b)
    dec_boundary = X * w + b;
    classifs = sign(dec_boundary);
    
    len = size(classifs(:,1));
    count = 0;
    for i = 1:len
        if (classifs(i) == labels(i))
            count = count + 1;
        end
    end
    correctly_classified = count/len(:,1);
    disp(['Of ' num2str(len(:,1)) ', ' num2str(count) ' points were correctly classified.']);
    disp(['Correctly classified? [0, 1]: ' num2str(correctly_classified)]);
end

% opt_point: calculates the current objective function value and returns it
function opt_point = calc_opt(w)
    opt_point = norm(w)/2;
end

% svmdistances: calculates the distance between the support vectors and 
% the found optimal hyperplane
% returns the distance (width) between the hyperplanes
% displays the distance to the positive class,
% distance to the negative class, and width
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