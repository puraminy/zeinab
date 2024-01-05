

% Extract features (X) and target variable (Y)
X = data(:, 2:6);  % Assuming the first 7 columns are features
Y = data(:, 1);    % Assuming the 8th column is the target variable

% Split the data into training (80%) and testing (20%) sets
rng('default');  % For reproducibility
cv = cvpartition(size(data, 1), 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);

X_train = X(idxTrain, :);
y_train = Y(idxTrain, :);

X_test = X(idxTest, :);
y_test = Y(idxTest, :);

% Normalize and scale the features if necessary
% You can use z-score normalization as an example
mu = mean(X_train);
sigma = std(X_train);

X_train = (X_train - mu) ./ sigma;
X_test = (X_test - mu) ./ sigma;

% Now you can use X_train, Y_train for training and X_test, Y_test for testing
% Your machine learning model training code goes here


% Define the neural network architecture
net = feedforwardnet([10, 5]);

% Load and preprocess your training data (X_train, y_train)

% Set training parameters
net.trainParam.epochs = 100;
net.trainParam.lr = 0.01;


% Assuming X_train and y_train are tables
X_train_transposed = rows2vars(X_train);
y_train_transposed = rows2vars(y_train);

% Assuming net is your neural network
net = train(net, X_train_transposed, y_train_transposed);


% Make predictions on new data
predictions = net(X_test');

