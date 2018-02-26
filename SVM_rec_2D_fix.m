function SVM = SVM_rec_2D_fix(train_pr,train_pos,dim1,dim2,rhat,pt,idx,leaf)

if(idx > leaf)
    SVM = [];
    return;
end
if(isempty(train_pos))    %bound the number of leaves
    SVM = struct('w',[],'b',1,'idx',idx,'l',2*idx,'r',2*idx+1);
    SVM = [SVM SVM_rec_2D_fix([],[],dim1,dim2/2,rhat,pt,idx*2+1,leaf)];
    SVM = [SVM SVM_rec_2D_fix([],[],dim1,dim2/2,rhat,pt,idx*2,leaf)]; 
    return;
end

class = train_pos(:,2) > dim2/2;
SVMModel = fitcsvm((train_pr/pt).^(-1/rhat),(class*2-1));
SVM = struct('w',SVMModel.Beta,'b',SVMModel.Bias,'idx',idx,'l',2*idx,'r',2*idx+1);
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
    SVM = [SVM SVM_rec_2D_fix(prpos,[],dim1,dim2/2,rhat,pt,idx*2+1,leaf)];
else 
    SVM = [SVM SVM_rec_2D_fix(prpos,[pospos(:,1),pospos(:,2)-dim2/2*ones(length(pospos(:,1)),1)],dim1,dim2/2,rhat,pt,idx*2+1,leaf)];
end

SVM = [SVM SVM_rec_2D_fix(prneg,posneg,dim1,dim2/2,rhat,pt,idx*2,leaf)];

return;