%% Setup
clear, clc, close all;

iris=load( 'data/iris.txt' );	% Load iris data
data = iris(:, 1:2);

%% (a) Plotting data (first 2 features)
close all;
FigHandle = figure;
plot(data(:,1), data(:,2), '.', 'MarkerSize', 24);
title('First two features of iris data');
xlabel('x_1');
ylabel('x_2');
saveas(FigHandle, ['Iris Data' '.png']);

%% (b) Run k-means on the data, for k = 5 and k = 20. Also try different initalization
close all;
initialization = {'random', 'farthest', 'k++'};
for i = 1:3
	for k = [5, 20]
		[z, c, cost] = kmeans(data, k, initialization{i});

		FigHandle = figure;
		plotClassify2D([], data, z); % Plot Clusters
		hold on;
		plot(c(:,1), c(:,2), 'kx', 'MarkerSize', 12, 'LineWidth', 2); % Plot Centroids
		hold off;
		
		titleStr = ['k-means, k = ', num2str(k), ', init = ', initialization{i}, ', cost = ', num2str(cost)];
		title(titleStr);
		xlabel('x_1');
		ylabel('x_2');
		
		saveas(FigHandle, [titleStr '.png']);
	end
end

%% (c) Run agglomerative clustering on the data, using single linkage...
...and then again using complete linkage, each with 5 and then 20 clusters.
close all;	
link = {'min', 'max'};
for i = 1:2
	for k = [5, 20]
		[z, join] = agglomCluster(data, k, link{i});

		FigHandle = figure;
		plotClassify2D([], data, z); % Plot Clusters
		
		titleStr = ['Agglomerative Clustering, k = ', num2str(k), ', link = ', link{i}];
		title(titleStr);
		xlabel('x_1');
		ylabel('x_2');
		
		saveas(FigHandle, [titleStr '.png']);
	end
end

%% (d) Run the EM Gaussian mixture model with 5 and 20 components.
close all;
initialization = {'random', 'farthest', 'k++'};
for i = 1:3
	for k = [5, 20]
		[z,T,soft,ll] = emCluster(data, k, initialization{i}); % Running EM GMM
		
		% -- Code to save figures as images -- %
		FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
		for iFig = 1:length(FigList)
		  FigHandle = FigList(iFig);
		  FigName   = ['k_', num2str(k), ' init_', initialization{i}];
		  set(0, 'CurrentFigure', FigHandle);
		  if(iFig == 1)
			  titleStr = ['Log Likelihood, k = ', num2str(k), ', init = ', initialization{i}];
			  xlabel('Iteration');
			  ylabel('Log Likelihood');
			  title(titleStr);
			  saveas(FigHandle, [titleStr '.png']);
		  else
			  titleStr = ['GMM, k = ', num2str(k), ', init = ', initialization{i}];
			  xlabel('x_1');
			  ylabel('x_2');
			  title(titleStr)
			  saveas(FigHandle, [titleStr '.png']);
		  end
		end
		% -- End of figure save -- %
	end
end


