function SVM = SVM_rec_1D_RBF(train_pr,train_pos,dim1,dim2,idx,leaf)

if(idx > leaf)    %5 levels at most
   SVM = []; 
   return;
end

if(isempty(train_pos))    %bound the number of leaves
    SVM = struct('clf',[],'idx',idx);
    SVM = [SVM SVM_rec_1D_RBF([],[],dim1/2,dim2,idx*2+1,leaf)];
    SVM = [SVM SVM_rec_1D_RBF([],[],dim1/2,dim2,idx*2,leaf)]; 
    return;
end

class = train_pos(:,1) > dim1/2;
if(isempty(class))
    SVM = [];
    return;
end

SVMModel = fitcsvm(train_pr,(class*2-1),'KernelFunction','rbf');
SVM = struct('clf',SVMModel,'idx',idx);

for i = 1:length(train_pr(1,:))
    pr = train_pr(:,i);
    prpos(:,i) = pr(class);
    prneg(:,i) = pr(~class);
end

for i = 1:length(train_pos(1,:))
    pos = train_pos(:,i);
    pospos(:,i) = pos(class);
    posneg(:,i) = pos(~class);
end

SVM = [SVM SVM_rec_1D_RBF(prpos,[pospos(:,1)-dim1/2*ones(length(pospos(:,1)),1),pospos(:,2)],dim1/2,dim2,idx*2+1,leaf)];

SVM = [SVM SVM_rec_1D_RBF(prneg,posneg,dim1/2,dim2,idx*2,leaf)];

return;