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
plotting(X_train, labels_train, w_cvx, b_cvx, 'CVX: ');
plotting(X_test, labels_test, w_cvx, b_cvx, 'CVX: ');
% result_cvx_train = eval(X_train, labels_train, w_cvx, b_cvx);
% checked and returns 1
result_cvx_test = eval(X_test, labels_test, w_cvx, b_cvx);

%% PROJECTED SUB GRADIENT SOLVER
disp(['-------- Solving with projSubGrad --------']); 
C = 1;
lr = 0.001;
max_num_iter = 10000;
epsilon = 0.001;
% could consider doing a more in depth parameter search but these gave
% reasonable answers
% CONSIDER: how do these parameters impact solving time
[w_grad, b_grad] = gradDesc(X_train, labels_train, C, lr, max_num_iter, epsilon);
plotting(X_train, labels_train, w_grad, b_grad, 'GD: ');
plotting(X_test, labels_test, w_grad, b_grad, 'GD: ');
result_grad_test = eval(X_test, labels_test, w_grad, b_grad);


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
        % minimize (norm(w))    
        minimize (square_pos(norm(w)))
        subject to
            labels .* (X * w + b) >= 1;
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
            % TODO CONSIDER THE STOP CRIT TO BE THE GRAD LESS THAN A VALUE
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

function [w, b] = Grad(X, labels, C, lr, max_num_iter, epsilon)
    disp(['Solving with Grad']);  

    % start function timing
    tStart = cputime;

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