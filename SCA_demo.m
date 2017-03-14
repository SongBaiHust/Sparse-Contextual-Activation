clc;
clear;
close all;

d = rand(10200, 10200, 'single');  % replace it with your input distance matrix
k1 = 4; % specify the first order of neighborhood set 
k2 = 2; % specify the second order of neighborhood set
[q_num, d_num] = size(d);
%% initialize the contextual feature f
[~, idx] = sort(d, 2, 'ascend');
fprintf('Jaccard re-ranking for k1 = %d...\n', k1);
f = zeros(size(d), 'single');
d = d./ repmat(max(d, [], 2), 1, size(d, 2));
for ii = 1:size(d, 1)
    idx_now = idx(ii, 1:k1);
    dis_now = d(ii, idx(ii, 1:k1));
    w = exp(-dis_now);
    f(ii, idx_now) = w/sum(w);
end
%% Local Enhancement
if k2 ~=1
    f1 = zeros(size(f), 'single');
    fprintf('Local consistency enhancement for k2 = %d...\n', k2);
    for ii = 1:size(f, 1)
        f1(ii, :) = single(sum(f(idx(ii, 1:k2), :)));
    end
    f = f1;
else
    fprintf('No local consistency enhancement...\n');
end
%% Indexing
fprintf('Buliding the inverted file...\n');
invIndex = cell(d_num, 1);
for ii = 1:d_num
    invIndex{ii} = find(f(:, ii) ~=0);
end
fprintf('Compute the distance matrix...\n');
dist = zeros(q_num, d_num, 'single');

for i = 1:q_num
    temp_min = zeros(1, q_num, 'single');
    indNonZero = find( f( i, : ) ~= 0 );
    indImages = invIndex( indNonZero );
    for j = 1 : length( indNonZero )
        temp_min( 1, indImages{j} ) = temp_min( 1, indImages{j} )...
            + single( min( f(i, indNonZero(j)), f(indImages{j}, indNonZero(j)) ) )';
    end
    dist(i, :) = bsxfun(@minus, 1, temp_min./(2*k2 - temp_min));
end
