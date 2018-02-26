function SVM = SVM_rec_1D_fix(train_pr,train_pos,dim1,dim2,rhat,pt,idx,leaf)

if(idx > leaf)    %5 levels at most
   SVM = []; 
   return;
end

if(isempty(train_pos))    %bound the number of leaves
    SVM = struct('w',[],'b',1,'idx',idx,'l',2*idx,'r',2*idx+1);
    SVM = [SVM SVM_rec_1D_fix([],[],dim1/2,dim2,rhat,pt,idx*2+1,leaf)];
    SVM = [SVM SVM_rec_1D_fix([],[],dim1/2,dim2,rhat,pt,idx*2,leaf)]; 
    return;
end

class = train_pos(:,1) > dim1/2;
if(isempty(class))
    SVM = [];
    return;
end

SVMModel = fitcsvm((train_pr/pt).^(-1/rhat),(class*2-1));
SVM = struct('w',SVMModel.Beta,'b',SVMModel.Bias,'idx',idx,'l',2*idx,'r',2*idx+1);

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

SVM = [SVM SVM_rec_1D_fix(prpos,[pospos(:,1)-dim1/2*ones(length(pospos(:,1)),1),pospos(:,2)],dim1/2,dim2,rhat,pt,idx*2+1,leaf)];

SVM = [SVM SVM_rec_1D_fix(prneg,posneg,dim1/2,dim2,rhat,pt,idx*2,leaf)];

return;