function statistical_significance_test(benchmark_dir)
%STATISTICAL_SIGNIFICANCE_TEST Compare detector F1 across repeated seeds.
% Run scripts/run_benchmark.py first, then call this function.

if nargin < 1
    benchmark_dir = fullfile('..', 'results', 'benchmark');
end
fold_path = fullfile(benchmark_dir, 'fold_metrics.csv');
if ~isfile(fold_path)
    error('Missing %s. Run scripts/run_benchmark.py first.', fold_path);
end

folds = readtable(fold_path);
models = unique(string(folds.model), 'stable');
seeds = unique(folds.seed, 'stable');
values = nan(numel(seeds), numel(models));
for model_index = 1:numel(models)
    for seed_index = 1:numel(seeds)
        match = folds.f1(string(folds.model) == models(model_index) & folds.seed == seeds(seed_index));
        if ~isempty(match)
            values(seed_index, model_index) = match(1);
        end
    end
end

if any(isnan(values), 'all')
    error('Each model must have an F1 score for each seed before paired comparison.');
end

[p, table_stats, stats] = friedman(values, 1, 'off');
fprintf('Friedman test p-value: %.6f\n', p);
disp('Friedman ANOVA output:');
disp(table_stats);

pairwise = multcompare(stats, 'Display', 'off');
pair_table = array2table(pairwise, 'VariableNames', {'ModelA', 'ModelB', 'LowerCI', 'Difference', 'UpperCI', 'pValue'});
pair_table.ModelA = models(pair_table.ModelA);
pair_table.ModelB = models(pair_table.ModelB);
disp(pair_table);

writetable(pair_table, fullfile(benchmark_dir, 'friedman_pairwise_f1.csv'));
end
