%% Setup
clear, clc, close all;
X = load('data/faces.txt'); % load face dataset
[m, n] = size(X); % Size of X (for later use)

%% Displaying first face
img = reshape(X(1,:),[24 24]); % convert vectorized datum to 24x24 image patch
FigHandle = imagesc(img); axis square; colormap gray; % display an image patch
titleStr = 'Example face 1';
saveas(FigHandle, [titleStr '.png']);
close all;
%% (a) Subtracting the mean of the face images to make data zero-mean.

mu = mean(X,1); % mean of each feature
X0 = bsxfun(@minus, X, mu); % Making X zero-mean

[U, S, V] = svd(X0); % Taking SVD of X

W = U*S; % Used later to compute approximation to X0

%% (b) Compute approximation to X0
K = 1:10;
MSErr = zeros(size(K));
for i = K
	X0h = W(:, 1:i)*V(:, 1:i)';
	MSErr(i) = mean(mean((X0-X0h).^2));
end

FigHandle = plot(K, MSErr, '-rx', 'DisplayName', 'MSErr');

xlabel('K');
ylabel('Mean Square Error');
titleStr = 'Mean Squared Error Vs. K';
title('Mean Squared Error Vs. K');
saveas(FigHandle, [titleStr '.png']);

close all;
%% (c) Display first few principal directions of the data

for j = 1:4
	alpha = 2*median(abs(W(:,j)));
	posPC = mu + alpha * V(:,j)';
	negPC = mu - alpha * V(:,j)';
	
	posIMG = reshape(posPC, [24, 24]);
	negIMG = reshape(negPC, [24, 24]);
	
	FigHandle = figure;
	imagesc(posIMG);
	colormap gray;
	axis square;
	titleStr = ['Principal Component ', num2str(j), ' (Positive)'];
	title(titleStr);
	saveas(FigHandle, [titleStr '.png']);
	close all;
	
	FigHandle = figure;
	imagesc(negIMG);
	colormap gray;
	axis square;
	titleStr = ['Principal Component ', num2str(j), ' (Negative)'];
	title(titleStr);
	saveas(FigHandle, [titleStr '.png']);
	close all;
end

%% (d) Latent Space methods
randidx = randperm(m);	% Randomize indices
idx = randidx(1:25);	% Get first 25 random indices of faces

FigHandle = figure; hold on; axis ij; colormap(gray);
range = max(W(idx,1:2)) - min(W(idx,1:2)); % find range of coordinates to be plotted
scale = [200 200]./range; % want 24x24 to be visible
for i=idx, imagesc(W(i,1)*scale(1),W(i,2)*scale(2), reshape(X(i,:),24,24)); end
axis square;
titleStr = 'Latent space visualisation';
title(titleStr);
saveas(FigHandle, [titleStr '.png']);
close all;

%% (e) Choose two faces and reconstruct them using K = 5, 10, 50 principal components
K = [5, 10, 50];
faceidx = [1000, 2000];

% Displaying original faces
FigHandle = figure;
imagesc(reshape(X(1000,:), [24, 24]));
colormap gray;
axis square;
title('Face 1000 Original');
saveas(FigHandle, ['Face 1000 Original' '.png']);
close all;

FigHandle = figure;
imagesc(reshape(X(2000,:), [24, 24]));
colormap gray;
axis square;
title('Face 2000 Original');
saveas(FigHandle, ['Face 2000 Original' '.png']);
close all;

% Reconstructing faces using K principal components and displaying
for k = K
	% Reconstructing all faces
	X0h = W(:, 1:k)*V(:, 1:k)';
	for f = faceidx
		% Displaying faces 1000 and 2000
		FigHandle = figure;
		imagesc(reshape(X0h(f,:), [24, 24]));
		colormap gray;
		axis square;
		titleStr = ['Face ', num2str(f), ', K=', num2str(k)];
		title(titleStr);
		saveas(FigHandle, [titleStr '.png']);
		close all;
	end
end


