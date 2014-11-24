
data = dlmread('parzenData.txt', ' ');
size(data);
parzenWindowClassifier(data, [0.5 1 0; 0.31 1.51 -0.5; -0.3 0.44 -0.1], 1.5)
