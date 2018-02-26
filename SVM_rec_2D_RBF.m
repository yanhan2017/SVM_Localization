function SVM = SVM_rec_2D_RBF(train_pr,train_pos,dim1,dim2,idx,leaf)

if(idx > leaf)
    SVM = [];
    return;
end
if(isempty(train_pos))    %bound the number of leaves
    SVM = struct('clf',[],'idx',idx);
    SVM = [SVM SVM_rec_2D_RBF([],[],dim1,dim2/2,idx*2+1,leaf)];
    SVM = [SVM SVM_rec_2D_RBF([],[],dim1,dim2/2,idx*2,leaf)]; 
    return;
end

class = train_pos(:,2) > dim2/2;
SVMModel = fitcsvm(train_pr,(class*2-1),'KernelFunction','rbf');
SVM = struct('clf',SVMModel,'idx',idx);
prpos = [];
prneg = [];
pospos = [];
posneg = [];

for i = 1:length(train_pr(1,:))
    pr = train_pr(:,i);
    if(~isempty(pr(class)))
        prpos = [prpos pr(class)];
    end
    
    if(~isempty(pr(~class)))
        prneg = [prneg pr(~class)];
    end
end

for i = 1:length(train_pos(1,:))
    pos = train_pos(:,i);
    if(~isempty(pos(class)))
        pospos = [pospos pos(class)];
    end
    if(~isempty(pos(~class)))
        posneg = [posneg pos(~class)];
    end
end

if (isempty(pospos))
    SVM = [SVM SVM_rec_2D_RBF(prpos,[],dim1,dim2/2,idx*2+1,leaf)];
else 
    SVM = [SVM SVM_rec_2D_RBF(prpos,[pospos(:,1),pospos(:,2)-dim2/2*ones(length(pospos(:,1)),1)],dim1,dim2/2,idx*2+1,leaf)];
end

SVM = [SVM SVM_rec_2D_RBF(prneg,posneg,dim1,dim2/2,idx*2,leaf)];

return;