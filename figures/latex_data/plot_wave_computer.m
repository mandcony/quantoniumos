% MATLAB Script: Wave Computer Benchmark Plotting
% Generates publication-quality figures from Python-exported CSV data

% Read data
data = readtable('wave_computer.csv', 'HeaderLines', 2);

% Extract columns
modes = data.Modes;
rft_mse = data.RFT_MSE;
fft_mse = data.FFT_MSE;

% Create figure
figure('Position', [100, 100, 800, 600]);

% Plot reconstruction error vs modes
semilogy(modes, rft_mse, 'o-', 'Color', [0.2, 0.6, 0.8], 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', [0.2, 0.6, 0.8], 'DisplayName', 'Graph \Phi-RFT (Wave Computer)');
hold on;
semilogy(modes, fft_mse, 's--', 'Color', [0.8, 0.2, 0.2], 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', [0.8, 0.2, 0.2], 'DisplayName', 'Standard FFT');

% Highlight the k=5 comparison
idx_5 = find(modes == 5);
if ~isempty(idx_5)
    semilogy(5, rft_mse(idx_5), 'o', 'Color', [0, 0.4, 0], 'MarkerSize', 15, 'LineWidth', 3, 'DisplayName', sprintf('RFT @ k=5: %.2e', rft_mse(idx_5)));
    semilogy(5, fft_mse(idx_5), 's', 'Color', [0.6, 0, 0], 'MarkerSize', 15, 'LineWidth', 3, 'DisplayName', sprintf('FFT @ k=5: %.2e', fft_mse(idx_5)));
end

% Formatting
xlabel('Number of Modes (Parameters)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Reconstruction MSE (Log Scale)', 'FontSize', 14, 'FontWeight', 'bold');
title('Wave Computer Efficiency: Modeling Fibonacci Graph Dynamics (N=64)', 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);
ylim([1e-12, 1e-2]);

% Add annotation box
if ~isempty(idx_5)
    ratio = fft_mse(idx_5) / rft_mse(idx_5);
    text_str = sprintf('\\Phi-RFT wins by %.1e\\times\nat 5 modes', ratio);
    annotation('textbox', [0.18, 0.75, 0.25, 0.1], 'String', text_str, ...
        'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [1, 1, 0.9], ...
        'EdgeColor', [0, 0.5, 0], 'LineWidth', 2);
end

% Save high-resolution PDF
print('../rft_wave_computer_matlab.pdf', '-dpdf', '-r300');
disp('✅ Saved: ../rft_wave_computer_matlab.pdf');

% Optional: Save PNG version
print('../rft_wave_computer_matlab.png', '-dpng', '-r300');
disp('✅ Saved: ../rft_wave_computer_matlab.png');
