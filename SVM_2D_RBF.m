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
dim1 = 20;
dim2 = 20;  %dimensions of the grid
tx = [1,1;dim1,1;1,dim2;dim1,dim2]; %tx position
ntx = length(tx(:,1));
nodes = 3;
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
    x = datasample(c,nsignal,1);   %signal sent
    for k = 1:ntx
        pr_dB(i,k) = pt - 10*r*log10(norm(position(i,:)-tx(k,:)));
        %pr_dB(i,:) = pr_dB(i,:) + S(i);
        omega = sqrt(N0_lin/2)*((randn(nsignal,1))+randn(nsignal,1)*1j);  %receiver noise
        y = x/rms(x)*sqrt(10.^(pr_dB(i,:)/10))+omega;   %signal received
        pr_est(i,:) = mean((abs(y)).^2)-N0_lin; %extract pr
    end
end
%%

figure(1)
plot(position(:,1),position(:,2),'o');
hold on
plot(tx(:,1),tx(:,2),'+');
hold off
legend('receivers','transmitters');
%%
node_list = [15 31 63 127];
rhat_list = [2.3 2.8 3.3 3.8 4.3 4.8];
error_list = [];

figure()
xlim([0 dim1]);
ylim([0 dim2]);
hold on

for n = 1%:length(rhat_list)
    %rhat = rhat_list(n)
    SVM1 = SVM_rec_1D_RBF(pr_est(1:train_count,:),position(1:train_count,:),dim1,dim2,1,nodes); %leaf = 2^level-1
    Sfield = fieldnames(SVM1);
    Scell = struct2cell(SVM1);
    sz = size(Scell);
    Scell = reshape(Scell,sz(1),[]);
    Scell = Scell';
    Scell = sortrows(Scell,2);
    Scell = reshape(Scell',sz);
    SVM1 = cell2struct(Scell,Sfield,1);  %sort SVM based on idx

    %% dim2 SVMs to classify along dim2
    SVM2 = [];

    for i = 1:nodes+1 %these many classes
        pr = [];
        pos = [];
        class = (ceil((position(1:train_count,1)-1)*(nodes+1)/(dim1-1)) == i);
        for k = 1:length(pr_est(1,:))
            prk = pr_est(1:train_count,k);
            pr(:,k) = prk(class);
        end
        for k = 1:length(position(1,:))
            posk = position(1:train_count,k);
            pos(:,k) = posk(class);
        end
        temp = SVM_rec_2D_RBF(pr,pos,dim1,dim2,1,nodes);
        Sfield = fieldnames(temp);
        Scell = struct2cell(temp);
        sz = size(Scell);
        Scell = reshape(Scell,sz(1),[]);
        Scell = Scell';
        Scell = sortrows(Scell,2);
        Scell = reshape(Scell',sz);
        temp = cell2struct(Scell,Sfield,1);  %sort temp based on idx

        SVM2 = [SVM2 temp];
    end

    SVM2 = reshape(SVM2,length(SVM1),length(SVM2)/length(SVM1));
    
    %% map SVM results to location
    mapping = zeros(nodes+1,nodes+1,2);
    for i = 1:(nodes+1)
        for k = 1:(nodes+1)
            mapping(i,k,:) = [1+(dim1-1)/(nodes+1)*(i-0.5) 1+(dim2-1)/(nodes+1)*(k-0.5)];  
        end
    end

    %% find the location of training data
    loc_test = zeros(test_count,2);
    error = 0;
    
    for i = train_count+1:data_count
        idx1 = 1;
        while (idx1 <= nodes)
            if (isempty(SVM1(idx1).clf))
                    idx1 = idx1 * 2 + 1;
            else
                if(SVM1(idx1).clf.predict(pr_est(i,:)) < 0)
                    idx1 = idx1 * 2;
                else
                    idx1 = idx1 * 2 + 1;
                end
            end
        end
        idx1 = idx1 - length(SVM1);
        idx2 = 1;
        while (idx2 <= nodes)    %leaves of SVM1
            if (isempty(SVM2(idx2,idx1).clf))
                    idx2 = idx2 * 2;
            else
                if(SVM2(idx2,idx1).clf.predict(pr_est(i,:)) < 0)
                    idx2 = idx2 * 2;
                else
                    idx2 = idx2 * 2 + 1;
                end
            end
        end
        idx2 = idx2 - length(SVM2(:,1));
        
        loc_test(i,:) = mapping(idx1,idx2,:);
        if(i <= train_count+35)
           text(position(i,1),position(i,2),['(' num2str(loc_test(i,:)) ')']); 
        end
        if(isnan(int16(mapping(idx1,idx2,:))))
           loc_test(i,:) = [dim1/2,dim2/2];
        end
        error = error + norm(loc_test(i,:)-position(i,:))^2;
    end
    error_ms = sqrt(error/test_count);
    error_list = [error_list error_ms];
    display(error_ms)
end
