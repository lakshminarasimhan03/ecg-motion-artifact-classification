clearvars; close all; clc;

files = {'chaexp1ndra_mp36.txt', 'chandeaexp2_mp36.txt', ...
         'chandraexp3_mp36.txt', 'chamdraexp4_mp36.txt'};
labels = {'Normal', 'Deep Breathing', 'Hand Movement', 'Walking'};

fs = 500;
win_len = 3*fs;

features = [];
class_labels = [];
all_segments = {};

fprintf('=== ECG CLASSIFICATION ===\n\n');

for i = 1:length(files)
    fprintf('[%d/%d] %s (%s)\n', i, length(files), files{i}, labels{i});
    
    if exist(files{i},'file')==2
        filepath = files{i};
    else
        listing = dir(fullfile(pwd,'**',files{i}));
        if isempty(listing)
            fprintf('  ERROR: File not found - SKIP\n\n');
            continue;
        else
            filepath = fullfile(listing(1).folder, listing(1).name);
        end
    end
    
    try
        fid = fopen(filepath, 'r');
        fgetl(fid);
        fgetl(fid);
        data = textscan(fid, '%f %f %f', 'Delimiter', '\t');
        fclose(fid);
        
        ecg = data{2};
        fprintf('  Samples: %d (%.1f sec)\n', length(ecg), length(ecg)/fs);
        
    catch ME
        if exist('fid', 'var') && fid ~= -1
            fclose(fid);
        end
        fprintf('  ERROR: %s - SKIP\n\n', ME.message);
        continue;
    end
    
    ecg(isnan(ecg) | isinf(ecg)) = 0;
    ecg = detrend(ecg);
    
    b = fir1(100, [0.5 50]/(fs/2), 'bandpass');
    ecg = filtfilt(b, 1, ecg);
    
    nWins = floor(length(ecg)/win_len);
    fprintf('  Windows: %d\n', nWins);
    
    if nWins == 0
        fprintf('  ERROR: Signal too short - SKIP\n\n');
        continue;
    end
    
    for w = 1:nWins
        seg = ecg((w-1)*win_len+1 : w*win_len);
        
        td = extract_time_features(seg);
        fd = extract_freq_features(seg, fs);
        
        if any(isnan([td fd])) || any(isinf([td fd]))
            continue;
        end
        
        features = [features; [td fd]];
        class_labels = [class_labels; i];
        all_segments{end+1} = seg;
    end
    
    fprintf('  Features: %d\n\n', nWins);
end

if isempty(features)
    error('No data processed!');
end

fprintf('=== SUMMARY ===\n');
fprintf('Total samples: %d\n', size(features,1));
fprintf('Features: %d\n\n', size(features,2));

for i = 1:length(labels)
    count = sum(class_labels == i);
    fprintf('  %d. %-20s: %3d (%.1f%%)\n', i, labels{i}, count, 100*count/length(class_labels));
end
fprintf('\n');

mu = mean(features, 1);
sigma = std(features, [], 1);
sigma(sigma==0) = 1;
features_norm = (features - mu) ./ sigma;

fprintf('=== CLASSIFICATION ===\n');

k_folds = min(5, min(histcounts(class_labels)));
cv = cvpartition(class_labels, 'KFold', k_folds, 'Stratify', true);

all_true = [];
all_pred = [];
fold_acc = zeros(k_folds, 1);

for k = 1:k_folds
    trIdx = cv.training(k);
    teIdx = cv.test(k);
    
    Xtr = features_norm(trIdx, :);
    Ytr = class_labels(trIdx);
    Xte = features_norm(teIdx, :);
    Yte = class_labels(teIdx);
    
    t = templateSVM('KernelFunction', 'gaussian', 'KernelScale', 'auto', ...
                    'BoxConstraint', 10);
    Mdl = fitcecoc(Xtr, Ytr, 'Learners', t);
    
    Ypred = predict(Mdl, Xte);
    
    all_true = [all_true; Yte];
    all_pred = [all_pred; Ypred];
    
    fold_acc(k) = mean(Ypred == Yte) * 100;
    fprintf('Fold %d: %.2f%%\n', k, fold_acc(k));
end

overall_acc = mean(all_true == all_pred) * 100;

fprintf('\n=== RESULTS ===\n');
fprintf('Accuracy: %.2f%% (+/- %.2f%%)\n\n', mean(fold_acc), std(fold_acc));

C = confusionmat(all_true, all_pred);
fprintf('Confusion Matrix:\n');
fprintf('%15s', '');
for i = 1:length(labels)
    fprintf('%15s', labels{i});
end
fprintf('\n');
for i = 1:length(labels)
    fprintf('%15s', labels{i});
    for j = 1:length(labels)
        fprintf('%15d', C(i,j));
    end
    fprintf('\n');
end
fprintf('\n');

for i = 1:length(labels)
    if sum(C(i,:)) > 0
        acc = C(i,i) / sum(C(i,:)) * 100;
        fprintf('%-20s: %.1f%%\n', labels{i}, acc);
    end
end

figure('Position', [50 50 1400 700]);

subplot(1,2,1);
imagesc(C);
colormap(flipud(gray));
colorbar;
set(gca, 'XTick', 1:length(labels), 'XTickLabel', labels, ...
         'YTick', 1:length(labels), 'YTickLabel', labels);
xlabel('Predicted');
ylabel('True');
title(sprintf('Confusion Matrix\nAccuracy: %.2f%%', overall_acc));
axis square;
for i = 1:length(labels)
    for j = 1:length(labels)
        text(j, i, num2str(C(i,j)), 'HorizontalAlignment', 'center', ...
             'Color', 'r', 'FontWeight', 'bold', 'FontSize', 12);
    end
end

subplot(1,2,2);
class_acc = diag(C) ./ sum(C,2) * 100;
bar(class_acc);
set(gca, 'XTickLabel', labels, 'XTick', 1:length(labels));
xtickangle(45);
ylabel('Accuracy (%)');
title('Per-Class Accuracy');
grid on;
ylim([0 105]);
for i = 1:length(class_acc)
    text(i, class_acc(i)+2, sprintf('%.1f%%', class_acc(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

figure('Position', [100 100 1400 900]);

n_per_class = 3;
plot_idx = 1;

for cls = 1:length(labels)
    cls_indices = find(class_labels == cls);
    
    if isempty(cls_indices)
        continue;
    end
    
    n_plot = min(n_per_class, length(cls_indices));
    
    for j = 1:n_plot
        if plot_idx > 12
            break;
        end
        
        subplot(4, 3, plot_idx);
        
        seg_idx = cls_indices(j);
        signal = all_segments{seg_idx};
        pred_cls = all_pred(seg_idx);
        
        t = (0:length(signal)-1) / fs;
        plot(t, signal, 'b-', 'LineWidth', 1.2);
        grid on;
        
        if cls == pred_cls
            clr = [0 0.6 0];
            status = 'CORRECT';
        else
            clr = [0.9 0 0];
            status = sprintf('WRONG (Pred: %s)', labels{pred_cls});
        end
        
        title(sprintf('%s\n%s', labels{cls}, status), 'Color', clr, 'FontWeight', 'bold');
        xlabel('Time (s)');
        ylabel('ECG (V)');
        xlim([0 max(t)]);
        
        plot_idx = plot_idx + 1;
    end
end

sgtitle('ECG Signal Classification Examples', 'FontSize', 14, 'FontWeight', 'bold');

function feats = extract_time_features(signal)
    signal = double(signal(:));
    signal = signal - mean(signal);
    
    feats = zeros(1, 10);
    
    feats(1) = std(signal);
    feats(2) = rms(signal);
    feats(3) = max(signal) - min(signal);
    feats(4) = mean(abs(signal));
    feats(5) = skewness(signal);
    feats(6) = kurtosis(signal);
    
    sgn = sign(signal);
    sgn(sgn==0) = 1;
    feats(7) = sum(abs(diff(sgn)) > 0) / length(signal);
    
    feats(8) = sum(abs(diff(signal))) / length(signal);
    
    env = abs(hilbert(signal));
    feats(9) = mean(env);
    feats(10) = std(env);
end

function feats = extract_freq_features(signal, fs)
    N = length(signal);
    window_len = min(N, 1024);
    noverlap = round(window_len * 0.5);
    
    [pxx, f] = pwelch(signal, hamming(window_len), noverlap, window_len, fs);
    
    P = pxx(:);
    F = f(:);
    Ptotal = sum(P);
    
    feats = zeros(1, 7);
    
    if Ptotal == 0 || isnan(Ptotal) || isinf(Ptotal)
        return;
    end
    
    feats(1) = sum(F .* P) / Ptotal;
    
    cumP = cumsum(P);
    med_idx = find(cumP >= 0.5*Ptotal, 1, 'first');
    if ~isempty(med_idx)
        feats(2) = F(med_idx);
    end
    
    [~, peak_idx] = max(P);
    feats(3) = F(peak_idx);
    
    band1 = get_bandpower(P, F, 0.1, 5);
    band2 = get_bandpower(P, F, 5, 15);
    band3 = get_bandpower(P, F, 15, 40);
    band4 = get_bandpower(P, F, 40, 100);
    
    total_band = band1 + band2 + band3 + band4;
    if total_band > 0
        feats(4) = band1 / total_band;
        feats(5) = band2 / total_band;
        feats(6) = band3 / total_band;
        feats(7) = band4 / total_band;
    end
end

function bp = get_bandpower(P, F, fmin, fmax)
    idx = (F >= fmin) & (F <= fmax);
    if sum(idx) < 2
        bp = 0;
        return;
    end
    bp = trapz(F(idx), P(idx));
    if isnan(bp) || isinf(bp)
        bp = 0;
    end
end
