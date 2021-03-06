function [] = myProject()
	close all;

    addpath('libsvm-3.14/windows');
    data = dlmread('testData.txt', ';');
	size(data)
	trainSetClass = data(:,end);
	trainSetFeatures = data(:,1:end-1);
	K = length(unique(trainSetClass));
    
    figure;
    hold on;
    line= ['ob'; '*g'; '+c'; 'xr'; '>y'];
    for k=1:K
        idx = find (trainSetClass == k);
        plot (trainSetFeatures(idx,1),trainSetFeatures(idx,2),  line(k,:), 'LineWidth', 2);
    end
    hold off;

    disp('Discriminant analysis')
    prediction_Bayes = classify(trainSetFeatures, trainSetFeatures, trainSetClass, 'quadratic');
    misclassificationRate= sum(prediction_Bayes~=trainSetClass)/length(trainSetClass)
    
	disp('kNN')
% 	prediction_kNN = knnclassify(trainSetFeatures, trainSetFeatures, trainSetClass, 1);
    prediction_kNN = knnClassifier(trainSetFeatures, trainSetFeatures, trainSetClass, 1);
    misclassificationRate= sum(prediction_kNN~=trainSetClass)/length(trainSetClass)

    disp('LIBSVM')
	configStr = sprintf('-s 0 -t 1 -d 2 -r 1 -g 1 -c 100');
	net = svmtrain(trainSetClass, trainSetFeatures, configStr);
   	testSetClass = ones(size(trainSetFeatures,1), 1); %libSVM needs but does not really uses 
	[prediction_SVM] = svmpredict (testSetClass, trainSetFeatures, net);
	misclassificationRate= sum(prediction_SVM~=trainSetClass)/length(trainSetClass)
    
    
return