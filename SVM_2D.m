close all

N0 = -100;  %-100dBm
N0_lin = 10^(N0/10);    %for sigma
pt = 20;
pt_lin = 10^(pt/10);  %30dBm
r = 3.8;
rhat = 3.2;
dim1 = 64;
dim2 = 64;  %dimensions of the grid
nodes = 63;
tx = [1,1;dim1,1;1,dim2;dim1,dim2]; %tx position
ntx = length(tx(:,1));
c = [1+1j;1-1j;-1+1j;-1-1j];
pr_dB = zeros(dim1*dim2,ntx);
pr_est = zeros(dim1*dim2,ntx);
nsignal = 1000;
rx = [];

for n = 1:dim1
    for m = 1:dim2
        check = 1;
        x = datasample(c,nsignal,1);   %signal sent
        for k = 1:ntx
            if(((n == tx(k,1)) && (m == tx(k,2))))
                check = 0;  %there is an overlap
                pr_est((n-1)*dim1+m,k) = inf;
            else
                %the amplitude that the kth rx receives if tx is at (n,m)
                pr_dB((n-1)*dim1+m,k) = pt - 10*r*log10(norm([n,m]-tx(k,:)));
                omega = sqrt(N0_lin/2)*((randn(nsignal,1))+randn(nsignal,1)*1j);  %receiver noise
                y = x/rms(x)*sqrt(10.^(pr_dB((n-1)*dim1+m,k)/10))+omega;   %signal received
                pr_est((n-1)*dim1+m,k) = mean((abs(y)).^2)-N0_lin; %extract pr
            end
        end
        if (check)
            rx = [rx; n,m];
        end
    end
end

figure(1)
plot(tx(:,1),tx(:,2),'+');
hold on
plot(rx(:,1),rx(:,2),'o');
hold off
legend('transmitters','receivers');

error_list = [];
node_list = [15 31 63 127];
rhat_list = [1.3];%[2.3 2.8 3.3 3.8 4.3 4.8];
test_count = 1000;
test_positions = [rand(test_count,1)*(dim1-1)+1 rand(test_count,1)*(dim2-1)+1];
test_pr_dB = zeros(test_count,ntx);
test_pr_est = zeros(test_count,ntx);
for i = 1:test_count %generate these data
    distance = zeros(ntx,1);
    k = 1;
    x = datasample(c,nsignal,1);   %signal sent
    while(k <= ntx)
        if(((test_positions(i,1) == tx(k,1)) && (test_positions(i,2) == tx(k,2))))
            test_pr_est(i,k) = inf;
        else
            %the amplitude that the kth rx receives if tx is at (n,m)
            test_pr_dB(i,k) = pt - 10*r*log10(norm(test_positions(i,:)-tx(k,:)));
            omega = sqrt(N0_lin/2)*((randn(nsignal,1))+randn(nsignal,1)*1j);  %receiver noise
            y = x/rms(x)*sqrt(10.^(test_pr_dB(i,k)/10))+omega;   %signal received
            test_pr_est(i,k) = mean((abs(y)).^2)-N0_lin; %extract pr
        end
        k = k + 1;
    end
end

for n = 1:length(rhat_list)
    %nodes = node_list(n)
    rhat = rhat_list(n)
    figure()
    xlim([0 dim1]);
    ylim([0 dim2]);
    %SVM to classify along dim1
    SVM1 = SVM_rec_2D(pr_est,dim1,dim2,rhat,pt_lin,1,nodes);
    Sfield = fieldnames(SVM1);
    Scell = struct2cell(SVM1);
    sz = size(Scell);
    Scell = reshape(Scell,sz(1),[]);
    Scell = Scell';
    Scell = sortrows(Scell,3);
    Scell = reshape(Scell',sz);
    SVM1 = cell2struct(Scell,Sfield,1);  %sort SVM based on idx

    %dim2 SVMs to classify along dim2
    SVM2 = [];
    for i = 1:length(SVM1)+1
        %taking i*dim1 at a time
        temp = SVM_rec_1D(pr_est(((i-1)*dim2+1):(i*dim2),:),dim2,1,rhat,pt_lin,1,nodes);
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

    %% evaluate performance on different rhat and random positions
    error = 0;

    for k = 1:test_count
           idx1 = 1;
            while (idx1 <= nodes)
                if (length(SVM1(idx1).w) < 4)
                    if(SVM1(idx1).b < 0)
                    %if(SVM1(idx1).b < 0)
                        idx1 = idx1 * 2;
                    else
                        idx1 = idx1 * 2 + 1;
                    end
                else
                    if((test_pr_est(k,:)/pt_lin).^(-1/rhat)*SVM1(idx1).w+SVM1(idx1).b < 0)
                    %if((test_pr_est(k,:))*SVM1(idx1).w+SVM1(idx1).b < 0)
                        idx1 = idx1 * 2;
                    else
                        idx1 = idx1 * 2 + 1;
                    end
                end
                
            end
            idx1 = idx1 - length(SVM1);
            idx2 = 1;
            while (idx2 <= nodes)    %leaves of SVM1
                if (length(SVM2(idx2,idx1).w) < 4)
                    if(SVM2(idx2,idx1).b < 0)
                    %if(SVM2(idx2,idx1).b < 0)
                        idx2 = idx2 * 2;
                    else
                        idx2 = idx2 * 2 + 1;
                    end
                else
                    if((test_pr_est(k,:)/pt_lin).^(-1/rhat)*SVM2(idx2,idx1).w+SVM2(idx2,idx1).b < 0)
                    %if((test_pr_est(k,:))*SVM2(idx2,idx1).w+SVM2(idx2,idx1).b < 0)
                        idx2 = idx2 * 2;
                    else
                        idx2 = idx2 * 2 + 1;
                    end
                end
            end
            idx2 = idx2 - length(SVM2(:,1));

            error = error + norm(test_positions(k,:) - [dim1/(nodes+1)*(idx1),dim2/(nodes+1)*(idx2)])^2;
            if(k <= 15)
                text(test_positions(k,1),test_positions(k,2),['(' num2str(idx1*dim1/(nodes+1)) ',' num2str(idx2*dim2/(nodes+1)) ')']);
            end
    end
    sqrt(error/test_count)
    error_list = [error_list sqrt(error/test_count)];
end
figure()
plot(rhat_list,error_list,'-o')
grid on

% figure()
% xlim([0 dim1]);
% ylim([0 dim2]);
% for k = 1:35
%     idx1 = 1;
%     while (idx1 <= nodes)    %leaves of SVM1
%         if(((test_pr(k,:)-sigma2)/pt).^(-1/rhat)*SVM1(idx1).w+SVM1(idx1).b < 0)
%             idx1 = idx1 * 2;
%         else
%             idx1 = idx1 * 2 + 1;
%         end
%     end
%     idx1 = idx1 - nodes;
%     idx2 = 1;
%     while (idx2 <= nodes)    %leaves of SVM1
%         if(((test_pr(k,:)-sigma2)/pt).^(-1/rhat)*SVM2(idx2,idx1).w+SVM2(idx2,idx1).b < 0)
%             idx2 = idx2 * 2;
%         else
%             idx2 = idx2 * 2 + 1;
%         end
%     end
%     idx2 = idx2 - nodes;
%     text(test_positions(k,1),test_positions(k,2),['(' num2str(idx1) ',' num2str(idx2) ')']);
% end

%%
% group = zeros(dim1,1);
% for i=1:dim1
%     idx = 1;
%     while (idx <= 2^(floor(log2(length(SVM1)+1)))-1)    %leaves of SVM1
%         %((pr((i-1)*dim2+1,:)-sigma2)/pt).^(-1/rhat)*SVM1(idx).w+SVM1(idx).b;
%         if(((pr((i-1)*dim2+3,:)-sigma2)/pt).^(-1/rhat)*SVM1(idx).w+SVM1(idx).b < 0)
%             %pr((i-1)*dim2+n,:) to test nth column
%             idx = idx * 2;
%         else
%             idx = idx * 2 + 1;
%         end
%     end
%     group(i) = (idx);
% end
% figure(2);
% plot(group,'o');

% group = zeros(dim1,1);
% col = 1;
% for i=1:dim2
%     idx = 1;
%     while (idx <= nodes)    %leaves of SVM1
%         %((pr((i-1)*dim2+1,:)-sigma2)/pt).^(-1/rhat)*SVM1(idx).w+SVM1(idx).b;
%         if(((pr((col-1)*dim2+i,:)-sigma2)/pt).^(-1/rhat)*SVM2(idx,col).w+SVM2(idx,col).b < 0)
%             %pr((i-1)*dim2+n,:) to test nth column
%             idx = idx * 2;
%         else
%             idx = idx * 2 + 1;
%         end
%     end
%     group(i) = (idx);
% end
% figure(3);
% plot(group,'o');
% 
% %% print estimated position using training data
% figure(4)
% xlim([0 dim1]);
% ylim([0 dim2]);
% title('rhat = 4.5')
% result = zeros(dim1,dim2,2);
% error_count = 0;
% mapping = zeros(dim1,dim2,2);
% error = zeros(dim1,dim2);
% for n = 1:dim1
%     for m = 1:dim2
%         idx1 = 1;
%         while (idx1 <= nodes)    %leaves of SVM1
%             if(((pr((n-1)*dim2+m,:)-sigma2)/pt).^(-1/rhat)*SVM1(idx1).w+SVM1(idx1).b < 0)
%                 idx1 = idx1 * 2;
%             else
%                 idx1 = idx1 * 2 + 1;
%             end
%         end
%         idx1 = idx1 - nodes;
%         idx2 = 1;
%         while (idx2 <= nodes)    %leaves of SVM1
%             if(((pr((n-1)*dim2+m,:)-sigma2)/pt).^(-1/rhat)*SVM2(idx2,idx1).w+SVM2(idx2,idx1).b < 0)
%                 idx2 = idx2 * 2;
%             else
%                 idx2 = idx2 * 2 + 1;
%             end
%         end
%         idx2 = idx2 - nodes;
%         result(n,m,:) = [idx1,idx2];
%         if(idx1 ~= n || idx2 ~= m)
%             error_count = error_count + 1;
%         end
%         text(n,m,['(' num2str(idx2) ',' num2str(idx1) ')']);
%         error(n,m) = norm([(idx1)*dim1/(nodes+1), (idx2)*dim2/(nodes+1)]-[n,m])^2;
%     end
% end
% 
% error_ms = mean(mean(error));
% display(error_ms)
% 
% %% tests with in_grid test data
% test_position = (1:dim2)';
% test_count = length(test_position);
% test_position = [ones(test_count,1) test_position];
% test_pr = zeros(test_count,nrx);
% for i = 1:test_count %generate these data
%     distance = zeros(nrx,1);
%     k = 1;
%     while(k <= nrx)
%         distance(k) = norm(test_position(i,:)-rx(k,:));
%         test_pr(i,k) = pt*(distance(k))^(-r)+sigma2;
%         k = k + 1;
%     end
% end
% 
% figure(5)
% xlim([0 dim1]);
% ylim([0 dim2]);
% hold on
% error = 0;
% for k = 1:test_count
%         idx1 = 1;
%         while (idx1 <= nodes)
%             if(((test_pr(k,:)-sigma2)/pt).^(-1/rhat)*SVM1(idx1).w+SVM1(idx1).b < 0)
%                 idx1 = idx1 * 2;
%             else
%                 idx1 = idx1 * 2 + 1;
%             end
%         end
%         idx1 = idx1 - length(SVM1);
%         idx2 = 1;
%         while (idx2 <= nodes)    %leaves of SVM1
%             if(((test_pr(k,:)-sigma2)/pt).^(-1/rhat)*SVM2(idx2).w+SVM2(idx2).b < 0)
%                 idx2 = idx2 * 2;
%             else
%                 idx2 = idx2 * 2 + 1;
%             end
%         end
%         idx2 = idx2 - length(SVM2(:,1));
%         
%         %text(position(k,1),position(k,2),num2str([dim1/(nodes+1)*(idx1-0.5),dim2/(nodes+1)*(idx2-0.5)]));
%         text(test_position(k,2),test_position(k,1),['(',num2str(idx1),',',num2str(idx2),')']);
%         error = error + norm(test_position(k,:) - [dim1/(nodes+1)*(idx1),dim2/(nodes+1)*(idx2)])^2;
% end
% error_rate = sqrt(error/test_count);
% display(error_rate)

%% test with random test data
% test_count = 1000;
% test_position = zeros(test_count,2);
% test_pr = zeros(test_count,nrx);
% for i = 1:test_count %generate these data
%     test_position(i,:) = [rand()*dim1,rand()*dim2];
%     distance = zeros(nrx,1);
%     k = 1;
%     while(k <= ntx) %make sure all rx are at least 1m away from tx
%         distance(k) = norm(test_position(i,:)-tx(k,:));
%         test_pr(i,k) = pt*(distance(k))^(-r)+sigma2;
%         k = k + 1;
%     end
% end
% 
% figure(6)
% xlim([0 dim1]);
% ylim([0 dim2]);
% hold on
% error = 0;
% for k = 1:test_count
%         idx1 = 1;
%         while (idx1 <= nodes)
%             if(((test_pr(k,:)-sigma2)/pt).^(-1/rhat)*SVM1(idx1).w+SVM1(idx1).b < 0)
%                 idx1 = idx1 * 2;
%             else
%                 idx1 = idx1 * 2 + 1;
%             end
%         end
%         idx1 = idx1 - length(SVM1);
%         idx2 = 1;
%         while (idx2 <= nodes)    %leaves of SVM1
%             if(((test_pr(k,:)-sigma2)/pt).^(-1/rhat)*SVM2(idx1).w+SVM2(idx1).b < 0)
%                 idx2 = idx2 * 2;
%             else
%                 idx2 = idx2 * 2 + 1;
%             end
%         end
%         idx2 = idx2 - length(SVM2(:,1));
%         
%         %text(position(k,1),position(k,2),num2str([dim1/(nodes+1)*(idx1-0.5),dim2/(nodes+1)*(idx2-0.5)]));
%         text(position(k,1),position(k,2),['(',num2str(idx1),',',num2str(idx2),')']);
%         error = error + norm(position(k,:) - [dim1/(nodes+1)*(idx1-0.5),dim2/(nodes+1)*(idx2-0.5)]);
% end
% error_rate = error/test_count;
% 
