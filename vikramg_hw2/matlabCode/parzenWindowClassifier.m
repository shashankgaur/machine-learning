function prediction = parzenWindowClassifier (trainData, testData, hh)
D = size(trainData,2)-1;                %Assuming last column is the corresponding class
prediction = zeros(size(testData,1),1); 
[Y,index] = sort(trainData(:,D+1));     %sort the trainData (just in case)
trainData = trainData(index,:);            
class = unique(trainData(:,D+1));       %find number of unique classes in the data
len_class = zeros(1,length(class)+1);
for i = 1:length(class)
    len_class(i+1) = find(trainData(:,D+1)==class(i),1,'last');
end
for i = 1:size(testData,1)
    val = zeros(1,length(class));
    for c = 1:length(class)             %break down the input data into classes
        for j = len_class(c)+1:len_class(c+1)
            diff = testData(i,:)-trainData(j,1:D);
            val(c) = val(c)+exp(-(diff* diff')/(2*hh*hh));
        end
    end
    prediction(i) = class(val==max(val)); %find the class corresponding to max
end
%TODO: less than 20 lines of code


