function analyze_results(results_dir)
%ANALYZE_RESULTS Read Python metric exports and create a publication-ready F1 chart.
% Usage: analyze_results('results')

if nargin < 1
    results_dir = fullfile('..', 'results');
end
metrics_path = fullfile(results_dir, 'metrics_summary.csv');
if ~isfile(metrics_path)
    error('Missing %s. Run scripts/run_experiment.py first.', metrics_path);
end

metrics = readtable(metrics_path);
metrics = sortrows(metrics, 'f1', 'descend');
figure('Color', 'w', 'Position', [100 100 900 480]);
bar(categorical(metrics.model), metrics.f1);
ylim([0 1]);
ylabel('F1 score');
title('Network anomaly detection comparison');
grid on;
xtickangle(25);

figures_dir = fullfile(results_dir, 'figures_matlab');
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end
exportgraphics(gcf, fullfile(figures_dir, 'f1_comparison_matlab.png'), 'Resolution', 220);

fprintf('Best F1 model: %s (%.3f)\n', string(metrics.model(1)), metrics.f1(1));
end
