filename = 'filename.csv';
X = csvread(filename);
Fs = 0.00378;                                       % Sampling frequency
T = 1/Fs;                                           % Sampling period
L = 1040;                                           % Length of signal
t = (0:L-1)*T;                                      % Time vector
Fn = Fs/2;                                          % Nyquist Frequency
FX = fft(X)/L;                                      % Fourier Transform
Fv = linspace(0, 1, fix(L/2)+1)*Fn;                 % Frequency Vector
Iv = 1:length(Fv);                                  % Index Vector
figure(1)
plot(Fv, abs(FX(Iv))*2)
grid
title('Fourier Transform Of Original Signal ‘X’')
xlabel('Frequency (Hz)')
ylabel('Amplitude')
FXdcoc = fft(X-mean(X))/L;                          % Fourier Transform (D-C Offset Corrected)
figure(2)
plot(Fv, abs(FXdcoc(Iv))*2)
grid
title('Fourier Transform Of D-C Offset Corrected Signal ‘X’')
xlabel('Frequency (Hz)')
ylabel('Amplitude')
[FXn_max,Iv_max] = max(abs(FXdcoc(Iv))*2);          % Get Maximum Amplitude, & Frequency Index Vector
Wp = 2*Fv(Iv_max)/Fn;                               % Passband Frequency (Normalised)
Ws = Wp*2;                                          % Stopband Frequency (Normalised)
Rp = 10;                                            % Passband Ripple (dB)
Rs = 30;                                            % Stopband Ripple (dB)
[n,Wn] = buttord(Wp,Ws,Rp,Rs);                      % Butterworth Filter Order
[b,a] = butter(n,Wn);                               % Butterworth Transfer Function Coefficients
[SOS,G] = tf2sos(b,a);                              % Convert to Second-Order-Section For Stability
figure(3)
freqz(SOS, 4096, Fs);                               % Filter Bode Plot
title('Lowpass Filter Bode Plot')
S = filtfilt(SOS,G,X);                              % Filter ‘X’ To Recover ‘S’
figure(4)
plot(t, X)                                          % Plot ‘X’
hold on
plot(t, S, '-r', 'LineWidth',1.5)                   % Plot ‘S’
hold off
grid
legend('‘X’', '‘S’', 'Location','N')
title('Original Signal ‘X’ & Uncorrupted Signal ‘S’')
xlabel('Time (sec)')
ylabel('Amplitude')