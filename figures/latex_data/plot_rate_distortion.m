% MATLAB Script: Rate-Distortion Plotting
% Generates publication-quality figures from Python-exported CSV data

% Read data
data = readtable('rate_distortion.csv', 'HeaderLines', 2);

% Extract columns
rate_dct = data.Rate_DCT;
dist_dct = data.Distortion_DCT;
rate_rft = data.Rate_RFT;
dist_rft = data.Distortion_RFT;
rate_hybrid = data.Rate_Hybrid;
dist_hybrid = data.Distortion_Hybrid;

% Create figure
figure('Position', [100, 100, 800, 600]);

% Plot rate-distortion curves
semilogy(rate_dct, dist_dct, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'DCT only');
hold on;
semilogy(rate_rft, dist_rft, 'r--s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '\Phi-RFT only');
semilogy(rate_hybrid, dist_hybrid, 'g-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Hybrid (Theorem 10)');

% Formatting
xlabel('Rate (Bits Per Pixel)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Distortion (MSE)', 'FontSize', 14, 'FontWeight', 'bold');
title('Rate-Distortion Analysis: Solving the ASCII Bottleneck', 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);

% Save high-resolution PDF
print('../rft_rate_distortion_matlab.pdf', '-dpdf', '-r300');
disp('✅ Saved: ../rft_rate_distortion_matlab.pdf');

% Optional: Save PNG version
print('../rft_rate_distortion_matlab.png', '-dpng', '-r300');
disp('✅ Saved: ../rft_rate_distortion_matlab.png');
