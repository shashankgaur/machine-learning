% input samples
X1=[rand(1,100);rand(1,100);ones(1,100)];   % class '+1'
X2=[rand(1,100);1+rand(1,100);ones(1,100)]; % class '-1'
X=[X1,X2];

% output class [-1,+1];
Y=[-ones(1,100),ones(1,100)];

% init weigth vector
w=[.5 .5 .5]';

% call perceptron
wtag=perceptron(X,Y,w);
% predict
ytag=wtag'*X;


% plot prediction over origianl data
figure;hold on
plot(X1(1,:),X1(2,:),'b.')
plot(X2(1,:),X2(2,:),'r.')

plot(X(1,ytag<0),X(2,ytag<0),'bo')
plot(X(1,ytag>0),X(2,ytag>0),'ro')
legend('class -1','class +1','pred -1','pred +1')