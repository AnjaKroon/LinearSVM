clear all;

%% DESCRIPTION
% Solving Linear SVM with two approaches
% Premade solver, CVX, outputs weights and bias of opt. hyperplane
% CVX uses a lot of CPU time but few iterations
% Low complexity algorithm, projected sub gradient solver
% ProjSubGrad has low CPU time but many more iterations
% After solving for w (2x1) and b (1x1), separating plane is plotted
% Visual results are presented for training and test sets
% An evaluation function also reports the accuracy

%% README
% Inputs: linear_svm.mat
% X_test            900x2             14400  double              
% X_train           100x2              1600  double              
% labels_test       900x1              7200  double              
% labels_train      100x1               800  double    

% Outputs:
% w_cvx                 2x1
% b_cvx                 1x1
% w_grad                 2x1
% b_grad                 1x1
% performance plots as figures
% CPU time displayed in cmd wd;

%% IMPORT DATA
load("linear_svm.mat", "labels_train", "labels_test", "X_test", "X_train")
whos

%% CVX SOLVER
disp(['-------- Solving with CVX solver --------']); 
[w_cvx, b_cvx] = solve_cvx(X_train, labels_train);
plotting(X_train, labels_train, w_cvx, b_cvx);
plotting(X_test, labels_test, w_cvx, b_cvx);
% result_cvx_train = eval(X_train, labels_train, w_cvx, b_cvx);
% checked and returns 1
result_cvx_test = eval(X_test, labels_test, w_cvx, b_cvx);

%% PROJECTED SUB GRADIENT SOLVER
disp(['-------- Solving with projSubGrad --------']); 
C = 1;
learning_rate = 0.001;
num_iterations = 10000;
epsilon = 0.001;
% could consider doing a more in depth parameter search but these gave
% reasonable answers
% CONSIDER: how do these parameters impact solving time
[w_grad, b_grad] = projSubGrad(X_train, labels_train, C, learning_rate, num_iterations, epsilon);
plotting(X_train, labels_train, w_grad, b_grad);
plotting(X_test, labels_test, w_grad, b_grad);
% result_grad_train = eval(X_train, labels_train, w_grad, b_grad);      
% checked and returns 1
result_grad_test = eval(X_test, labels_test, w_grad, b_grad);

% CONSIDER: also solving the primal and dual problems


%% SUBFUNCTIONS
function [w, b] = solve_cvx(X, labels)
    disp(['Solving with CVX solver']);      
    tStart = cputime;
    [r,c] = size(X);
    cvx_begin
        variables w(c) b;
        minimize (norm(w))      % try also with norm^2
        subject to
            labels .* (X * w + b) >= 1;
    cvx_end
    tEnd = cputime - tStart;
    disp(['Total CPU time: ' num2str(tEnd)]);   
end

function [w, b] = projSubGrad(X, labels, C, learning_rate, num_iterations, epsilon)
    disp(['Solving with projSubGrad']);  
    tStart = cputime;
    [r,c] = size(X);
    w = zeros(c,1); 
    b = 0;
    prev_loss = inf;
    for iteration = 1:num_iterations
        loss = 0;
        for i = 1:r
            if (labels(i) * (X(i,:) * w + b)) < 1
                % sub gradients calculated at current solution
                dw = (w - C * labels(i) * X(i,:)');
                db = (-C * labels(i));
                % updating with subgrad
                w = w - learning_rate * dw;
                b = b - learning_rate * db;
                loss = loss + 1 - labels(i) * (X(i,:) * w + b);
            else
                w = w - learning_rate * w;
            end
        end

        % Execute the projection onto a feasible set
        % Thus normalize the weight vector to be size 1 (if C = 1)
        w_norm = norm(w);
        if w_norm > 1 / sqrt(C)
            w = w / w_norm * (1 / sqrt(C));
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
    disp(['Max iterations obtained unless early stopping: ' num2str(num_iterations)]); 
end

function plotting(X, labels, w, b) 
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
    title(sprintf('%s Data Points and Optimal Separational Hyperplane (SVM)', curDataName));
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
    disp(['Percentage correctly classified: ' num2str(correctly_classified)]);
end