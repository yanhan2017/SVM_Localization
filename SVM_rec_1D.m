function SVM = SVM_rec_1D(pr,dim1,dim2,rhat,pt,idx,nodes)

if(dim1 < 2 || idx > nodes)    %5 levels at most
   SVM = []; 
   return;
end

len1 = round(dim1/2);
len2 = dim1-round(dim1/2);
SVMModel = fitcsvm((pr/pt).^(-1/rhat),[-ones(len1*dim2,1);ones(len2*dim2,1)]);%,'KernelFunction','rbf','KFold',5);
%SVMModel = fitcsvm(pr,[-ones(len1*dim2,1);ones(len2*dim2,1)]);
SVM = struct('w',SVMModel.Beta,'b',SVMModel.Bias,'idx',idx,'l',2*idx,'r',2*idx+1);

prpos = pr(1:round(dim1/2),:);
prneg = pr(round(dim1/2)+1:end,:);

SVM = [SVM SVM_rec_1D(prpos,round(dim1/2),dim2,rhat,pt,idx*2,nodes)];

SVM = [SVM SVM_rec_1D(prneg,dim1-round(dim1/2),dim2,rhat,pt,idx*2+1,nodes)];

return;

