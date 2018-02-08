%error vs. tree level
%plug training data into SVM to find the position of each leaf
%% 
close all
N0 = -100;  %-100dBm
N0_lin = 10^(N0/10);    %for sigma
pt = 20;
pt_lin = 10^(pt/10);  %20dBm
r = 3.8;
rhat = 3.2;
sigma_s = 8;     %8dB, shadowing
x_c = 10;   %meter
dim1 = 100;
dim2 = 100;  %dimensions of the grid
tx = [1,1;dim1,1;1,dim2;dim1,dim2]; %tx position
ntx = length(tx(:,1));
nodes = 63;
c = [1+1j;1-1j;-1+1j;-1-1j];
%% 

data_count = 10000;
train_count = 8000;
test_count = data_count - train_count;  %use these numbers of data for training and testing
position = [rand(data_count,1)*(dim1-1)+1 rand(data_count,1)*(dim2-1)+1];  %array to store locations of rx
pr_dB = zeros(data_count, ntx); %array to store received signal strength from each tx
pr_est = zeros(data_count,ntx);
nsignal = 1000;

%R = zeros(train_count, train_count);    %correlation matrix for shadowing
% for i = 1:nrx
%     for k = 1:nrx
%         R(i,k) = exp(-norm(position(i,:)-position(k,:))/x_c);
%     end
% end
% R = sigma_s^2*R;
% sqrtR = sqrtm(R);
%%
for i = 1:data_count
    %S = sqrtR*randn(nrx,1);
    S = 0;
    for k = 1:ntx
        pr_dB(i,k) = pt - 10*r*log10(norm(position(i,:)-tx(k,:)));
    end
    %pr_dB(i,:) = pr_dB(i,:) + S(i);
    x = datasample(c,1000,1);   %signal sent
    omega = sqrt(N0_lin/2)*((randn(1000,1))+randn(1000,1)*1j);  %receiver noise
    y = x/rms(x)*sqrt(10.^(pr_dB(i,:)/10))+omega;   %signal received
    pr_est(i,:) = mean((abs(y)).^2)-N0_lin; %extract pr
end
%%

figure(1)
plot(position(:,1),position(:,2),'o');
hold on
plot(tx(:,1),tx(:,2),'+');
hold off
legend('receivers','transmitters');
%%
% class = position(:,1) > dim1/2;
% class = class*2 - 1;    %in class 1 if on the left half, in class 2 if on the right half
% SVMModel = fitcsvm(pr_est(1:train_count,:),class(1:train_count));
% 
% class = position(:,1) > dim1/2;
% prpos = [];
% prneg = [];
% pospos = [];
% posneg = [];
% for i = 1:length(pr_est(1,:))
%     pr = pr_est(1:train_count,i);
%     prpos(:,i) = pr(class(1:train_count));
%     prneg(:,i) = pr(~class(1:train_count));
% end
% 
% for i = 1:length(position(1,:))
%     pos = position(1:train_count,i);
%     pospos(:,i) = pos(class(1:train_count));
%     posneg(:,i) = pos(~class(1:train_count));
% end
% 
% class1 = posneg(:,1) > dim1/4;
% class2 = pospos(:,1) > dim1*3/4;
% class1 = class1*2-1;
% class2 = class2*2-1;
% SVMModel1 = fitcsvm(prneg,class1);
% SVMModel2 = fitcsvm(prpos,class2);
% 
% error_count = 0;
% for i = 1:train_count
%     if((pr_est(i,:)*SVMModel.Beta + SVMModel.Bias) < 0)
%         if((pr_est(i,:)*SVMModel1.Beta + SVMModel1.Bias) < 0 && position(i,1) > dim1/4)
%             error_count = error_count + 1;
%         elseif((pr_est(i,:)*SVMModel1.Beta + SVMModel1.Bias) > 0 && position(i,1) < dim1/4)
%             error_count = error_count + 1;
%         end
%     else
%         if((pr_est(i,:)*SVMModel2.Beta + SVMModel2.Bias) < 0 && position(i,1) > dim1*3/4)
%             error_count = error_count + 1; 
%         elseif((pr_est(i,:)*SVMModel2.Beta + SVMModel2.Bias) > 0 && position(i,1) < dim1*3/4)
%             error_count = error_count + 1;
%         end
%     end
% end
% train_error_rate = error_count/train_count;
% 
% error_count = 0;
% for i = train_count+1 : data_count
%     if((pr_est(i,:)*SVMModel.Beta + SVMModel.Bias)*class(i) < 0)
%        error_count = error_count + 1; 
%     end
% end
% test_error_rate = error_count/test_count;
% display(train_error_rate);
% display(test_error_rate);
% 
% class1 = [];
% class2 = [];
% w = SVMModel1.Beta;
% b = SVMModel1.Bias;
% for i = 1:length(prneg)
%     if(prneg(i,:)*w+b < 0)
%         class1 = [class1; posneg(i,:)];
%     else
%         class2 = [class2; posneg(i,:)];
%     end
% end
% 
% figure(2)
% plot(class1(:,1),class1(:,2),'o')
% hold on
% plot(class2(:,1),class2(:,2),'o')
% hold off
%%
SVM1 = SVM_rec_1D_fix(pr_est(1:train_count,:),position(1:train_count,:),dim1,dim2,1,nodes); %leaf = 2^level-1
Sfield = fieldnames(SVM1);
Scell = struct2cell(SVM1);
sz = size(Scell);
Scell = reshape(Scell,sz(1),[]);
Scell = Scell';
Scell = sortrows(Scell,3);
Scell = reshape(Scell',sz);
SVM1 = cell2struct(Scell,Sfield,1);  %sort SVM based on idx

%% dim2 SVMs to classify along dim2
SVM2 = [];
col = zeros(train_count,1);
for i = 1:train_count
    idx1 = 1;
    while (idx1 <= nodes)
        if(pr_est(i,:)*SVM1(idx1).w+SVM1(idx1).b < 0)
            idx1 = idx1 * 2;
        else
            idx1 = idx1 * 2 + 1;
        end
    end
    idx1 = idx1 - length(SVM1);
    col(i) = idx1;
end
for i = 1:nodes+1 %these many classes
    pr = [];
    pos = [];
%     class = (position(1:train_count,1) < i*dim1/(2^level)) & (position(1:train_count,1) > (i-1)*dim1/(2^level));
%     for k = 1:length(pr_est(1,:))
%         prk = pr_est(1:train_count,k);
%         pr(:,k) = prk(class);
%     end
%     for k = 1:length(position(1,:))
%         posk = position(1:train_count,k);
%         pos(:,k) = posk(class);
%     end
    class = col == i;
    for k = 1:length(pr_est(1,:))
        prk = pr_est(1:train_count,k);
        pr(:,k) = prk(class);
    end
    for k = 1:length(position(1,:))
        posk = position(1:train_count,k);
        pos(:,k) = posk(class);
    end
    temp = SVM_rec_2D_fix(pr,pos,dim1,dim2,1,nodes);
    Sfield = fieldnames(temp);
    Scell = struct2cell(temp);
    sz = size(Scell);
    Scell = reshape(Scell,sz(1),[]);
    Scell = Scell';
    Scell = sortrows(Scell,3);
    Scell = reshape(Scell',sz);
    temp = cell2struct(Scell,Sfield,1);  %sort temp based on idx

    SVM2 = [SVM2 temp];
end

SVM2 = reshape(SVM2,length(SVM1),length(SVM2)/length(SVM1));

%% map SVM results to location
group = zeros(train_count,2);
for i = 1:train_count
    idx1 = 1;
    while (idx1 <= nodes)
        if(pr_est(i,:)*SVM1(idx1).w+SVM1(idx1).b < 0)
            idx1 = idx1 * 2;
        else
            idx1 = idx1 * 2 + 1;
        end
    end
    idx1 = idx1 - length(SVM1);
    idx2 = 1;
    while (idx2 <= nodes)    %leaves of SVM1
        if(pr_est(i,:)*SVM2(idx2,idx1).w+SVM2(idx2,idx1).b < 0)
            idx2 = idx2 * 2;
        else
            idx2 = idx2 * 2 + 1;
        end
    end
    idx2 = idx2 - length(SVM2(:,1));
    group(i,:) = [idx1,idx2];
end

mapping = zeros(nodes+1,nodes+1,2);
for i = 1:(nodes+1)
    for k = 1:(nodes+1)
        class = (group(:,1) == i & group(:,2) == k);
        pos = [];
        for n = 1:length(position(1,:))
            posn = position(1:train_count,n);
            pos(:,n) = posn(class);
        end
        for n = 1:length(position(1,:))
            mapping(i,k,n) = mean(pos(:,n));
        end  
    end
end

%% find the location of training data
loc_test = zeros(test_count,2);
error = 0;
for i = train_count+1:data_count
    idx1 = 1;
    while (idx1 <= nodes)
        if(pr_est(i,:)*SVM1(idx1).w+SVM1(idx1).b < 0)
            idx1 = idx1 * 2;
        else
            idx1 = idx1 * 2 + 1;
        end
    end
    idx1 = idx1 - length(SVM1);
    idx2 = 1;
    while (idx2 <= nodes)    %leaves of SVM1
        if(pr_est(i,:)*SVM2(idx2,idx1).w+SVM2(idx2,idx1).b < 0)
            idx2 = idx2 * 2;
        else
            idx2 = idx2 * 2 + 1;
        end
    end
    idx2 = idx2 - length(SVM2(:,1));
    loc_test(i,:) = mapping(idx1,idx2);
    if(isnan(int16(mapping(idx1,idx2))))
        loc_test(i,:) = [dim1/2,dim2/2];
    end
    error = error + norm(loc_test(i,:)-position(i,:))^2;
end
error_ms = sqrt(error/test_count);
display(error_ms)
%% plot the grouping
% group = zeros(test_count,1);
% 
% for i = 1:train_count
%     idx = 1;
%     while (idx <= 2^(floor(log2(length(SVM1)+1)))-1)
%         if(pr_est(i,:)*SVM1(idx).w+SVM1(idx).b < 0)
%             idx = idx * 2;
%         else
%             idx = idx * 2 + 1;
%         end
%     end
%     group(i) = (idx);
% end
% 
% group = group - min(group) + 1;
% figure(3);
% xlim([0 dim1]);
% ylim([0 dim2]);
% hold on
% for i = 1:test_count
%     text(position(i,1),position(i,2),num2str(group(i)));
% end