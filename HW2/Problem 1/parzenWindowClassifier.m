function prediction = parzenWindowClassifier (training, test, hh)
d = size(training,2)-1;                
prediction = zeros(size(test,1),1); 
[s,index] = sort(training(:,d+1));     
training = training(index,:);            
class = unique(training(:,d+1));       
len_class = zeros(1,length(class)+1);
for i = 1:length(class)
    len_class(i+1) = find(training(:,d+1)==class(i),1,'last');
end
for i = 1:size(test,1)
    val = zeros(1,length(class));
    for c = 1:length(class)             
        for j = len_class(c)+1:len_class(c+1)
            diff = test(i,:)-training(j,1:d);
            val(c) = val(c)+exp(-(diff* diff')/(2*hh*hh));
        end
    end
    prediction(i) = class(val==max(val)); 
end

