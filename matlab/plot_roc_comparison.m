function plot_roc_comparison(results_dir)
%PLOT_ROC_COMPARISON Plot all ROC exports created by the Python pipeline.
% Usage: plot_roc_comparison('results')

if nargin < 1
    results_dir = fullfile('..', 'results');
end
files = dir(fullfile(results_dir, 'roc_*.csv'));
if isempty(files)
    error('No ROC CSV files found in %s.', results_dir);
end

figure('Color', 'w', 'Position', [100 100 700 560]);
hold on;
for index = 1:numel(files)
    curve = readtable(fullfile(files(index).folder, files(index).name));
    plot(curve.fpr, curve.tpr, 'LineWidth', 1.5, 'DisplayName', string(curve.model(1)));
end
plot([0 1], [0 1], '--k', 'DisplayName', 'Chance');
xlim([0 1]); ylim([0 1]); grid on;
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC comparison: network anomaly detectors');
legend('Location', 'southeast', 'Interpreter', 'none');

figures_dir = fullfile(results_dir, 'figures_matlab');
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end
exportgraphics(gcf, fullfile(figures_dir, 'roc_comparison_matlab.png'), 'Resolution', 220);
end
