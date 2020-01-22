function varargout = MONITORING(varargin)
%MONITORING MATLAB code file for MONITORING.fig
%      MONITORING, by itself, creates a new MONITORING or raises the existing
%      singleton*.
%
%      H = MONITORING returns the handle to a new MONITORING or the handle to
%      the existing singleton*.
%
%      MONITORING('Property','Value',...) creates a new MONITORING using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to MONITORING_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      MONITORING('CALLBACK') and MONITORING('CALLBACK',hObject,...) call the
%      local function named CALLBACK in MONITORING.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MONITORING

% Last Modified by GUIDE v2.5 25-Apr-2019 14:11:37

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @MONITORING_OpeningFcn, ...
    'gui_OutputFcn',  @MONITORING_OutputFcn, ...
    'gui_LayoutFcn',  [], ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before MONITORING is made visible.
function MONITORING_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   unrecognized PropertyName/PropertyValue pairs from the
%            command line (see VARARGIN)


% Choose default command line output for MONITORING
handles.output = hObject;
handles.setupdata.curFileName = [];

databaseFiles = dir('Spkmodel-*.mat');
databaseName = cell(1,length(databaseFiles));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% change here 10/14
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(databaseFiles)
    disp("hello world\n")
    for i = 1:length(databaseFiles)
        databaseName{i} = databaseFiles(i).name(1:end-4);
    end
    %     handles.database = load(databaseFiles(1).name);
    %     handles.database.no_of_fe = length(handles.database.gmm_models);
else
    handles.database.ubm = [];
    handles.database.gmm_models = [];
    handles.database.no_of_fe = [];
end
set(handles.pm_DataBase,'String',...
    databaseName)

global stopbit;
stopbit = 1; % continue

% % Update handles structure
guidata(hObject, handles);

% UIWAIT makes MONITORING wait for user response (see UIRESUME)
% uiwait(handles.figure1);



% --- Outputs from this function are returned to the command line.
function varargout = MONITORING_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure

varargout{1} = handles.output;
contents = cellstr(get(handles.pm_DataBase,'String'));
ncurDatabase = get(handles.pm_DataBase,'value');
dataBasefileName = [contents{ncurDatabase} '.mat'];
if ~exist( dataBasefileName,'file')
    warndlg('Default Feature file not found!','Error');
    handles.database.no_of_fe = 0;
else
    handles.database = load(dataBasefileName);
    handles.database.dataBasefileName = dataBasefileName;
    handles.database.no_of_fe = length(handles.database.gmm_models);
end
if isempty(handles.database.no_of_fe)
    handles.database.no_of_fe = 0;
    handles.database.name = char('');
end

if ~isempty(handles.database.gmm_models)
    speakerName = cell(1,length(handles.database.gmm_models));
    for i = 1:length(handles.database.gmm_models)
        speakerName{i} = handles.database.gmm_models{1, i}.name;
    end
end
set(handles.pm_SpeakerName,'String',...
    speakerName)
handles.speakerName = speakerName;
set(handles.text_DataBaseNumber,'String',handles.database.no_of_fe);
guidata(hObject, handles);


% --- Executes on button press in bt_MFCC.
function bt_MFCC_Callback(hObject, eventdata, handles)
% hObject    handle to bt_MFCC (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

Fs = handles.setupdata.Fs;
curSignal = handles.setupdata.curSignal;
% axes(handles.axes_Spectrogram)
% cla(gca)
% spectrumplot(curSignal,Fs)

def = {'1';'0';'8'};
prompt = {'Enter order of melcepst:';...
    'Enter low frequency f_l(kHz):';...
    'Enter high frequency f_h(kHz):'};
dlg_title = 'Input for feature extraction ';
num_lines = 1;
answer = inputdlg(prompt,dlg_title,num_lines,def);

if ~isempty(answer)
    norders = str2num(answer{1});
    fl = str2num(answer{2});
    fh = str2num(answer{3});
    fl = fl*1e3;
    fh = fh*1e3;
    if fh<=fl
        errordlg('f_h should be larger than f_l!')
        return
    elseif fh>Fs/2
        fh = Fs/2;
    end
else
    return
end

if norders == 1
    [mfccFeature,t,ci] = melcepst(curSignal,[fl fh Fs],'M0');
elseif norders == 2
    [mfccFeature,t,ci] = melcepst(curSignal,[fl fh Fs],'MD0');
elseif norders == 3
    [mfccFeature,t,ci] = melcepst(curSignal,[fl fh Fs],'MDd0');
end
axes(handles.axes_Feature)
cla(gca)
imagesc(t,ci,mfccFeature.');
axis('xy');
xlabel('Time (secs)');
ylabel('Mel-cepstrum coefficient');
% hcb = colorbar;
handles.mfccFeature = mfccFeature;
guidata(hObject,handles)

function spectrumplot(y,Fs)

% win_sz = pow2(floor(log2(0.03*Fs))-1);
win_sz = 512;
han_win = hanning(win_sz);      % ѡ������
nfft = win_sz;
nooverlap = win_sz - 20;
[S, F, T] = spectrogram(y, han_win, nooverlap, nfft, Fs);

imagesc(T, F/1e3, log10(abs(S)))
set(gca, 'YDir', 'normal')
xlabel('Time (secs)')
ylabel('Freq (kHz)')
title('short time fourier transform spectrum')
ylim([0,Fs/2/1e3])
% hcb = colorbar;
% ylabel(hcb, 'Power/Freq (dB/Hz)')


function [c,t,ci] = melcepst(s,f,w,nc,p,n,inc,fl,fh)
%MELCEPST Calculate the mel cepstrum of a signal C=(S,F,W,NC,P,N,INC,FL,FH)
%
%
% Simple use: (1) c=melcepst(s,fs)          % calculate mel cepstrum with 12 coefs, 256 sample frames
%			  (2) c=melcepst(s,fs,'e0dD')   % include log energy, 0th cepstral coef, delta and delta-delta coefs
%
% Inputs:
%     s	  speech signal
%     f  [fl*fs fh*fs fs] low frequency, highfrequency and sample rate in Hz (default 11025)
%     w   mode string (see below)
%     nc  number of cepstral coefficients excluding 0'th coefficient [default 12]
%     p   number of filters in filterbank [default: floor(3*log(fs)) =  approx 2.1 per ocatave]
%     n   length of frame in samples [default power of 2 < (0.03*fs)]
%     inc frame increment [default n/2]
%     fl  low end of the lowest filter as a fraction of fs [default = 0]
%     fh  high end of highest filter as a fraction of fs [default = 0.5]
%
%		w   any sensible combination of the following:
%
%               'R'  rectangular window in time domain
%				'N'	 Hanning window in time domain
%				'M'	 Hamming window in time domain (default)
%
%               't'  triangular shaped filters in mel domain (default)
%               'n'  hanning shaped filters in mel domain
%               'm'  hamming shaped filters in mel domain
%
%				'p'	 filters act in the power domain
%				'a'	 filters act in the absolute magnitude domain (default)
%
%               '0'  include 0'th order cepstral coefficient
%				'E'  include log energy
%				'd'	 include delta coefficients (dc/dt)
%				'D'	 include delta-delta coefficients (d^2c/dt^2)
%
%               'z'  highest and lowest filters taper down to zero (default)
%               'y'  lowest filter remains at 1 down to 0 frequency and
%			   	     highest filter remains at 1 up to nyquist freqency
%
%		       If 'ty' or 'ny' is specified, the total power in the fft is preserved.
%
% Outputs:	c     mel cepstrum output: one frame per row. Log energy, if requested, is the
%                 first element of each row followed by the delta and then the delta-delta
%                 coefficients.
%

% BUGS: (1) should have power limit as 1e-16 rather than 1e-6 (or possibly a better way of choosing this)
%           and put into VOICEBOX
%       (2) get rdct to change the data length (properly) instead of doing it explicitly (wrongly)

%      Copyright (C) Mike Brookes 1997
%      Version: $Id: melcepst.m 3497 2013-09-26 16:10:51Z dmb $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   http://www.gnu.org/copyleft/gpl.html or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<2
    f = [0 8e3 16e3];
end
flc = f(1);
fhc = f(2);
fs = f(3);
if nargin<3
    w = 'M';
end
if nargin<4
    nc = 12;
end
if nargin<5
    p = floor(3*log(fs)); end
if nargin<6
    n = pow2(floor(log2(0.03*fs)));
    %     n = pow2(floor(log2(0.01*fs)));
end
if nargin<9
    fh = 0.5;
    if nargin<8
        fl = 0;
        if nargin<7
            inc = floor(n/2);
        end
    end
end
fh = min((fhc-flc)/fs,fh);
if isempty(w)
    w = 'M';
end
if any(w =='R')
    z = enframe(s,n,inc);
elseif any (w=='N')
    z = enframe(s,hanning(n),inc);
else
    z = enframe(s,hamming(n),inc);
end
f = rfft(z.');
% idea test-----------------------
% only use the featurn on the frequency range >8kHz
nflc = fix(flc/(fs/n));
f(1:nflc,:) = [];
n = n - 2*nflc;
% -----------------------------------------------
[m,~,a,b] = melbankm(p,n,fs,fl,fh,w);
pw = f(a:b,:).*conj(f(a:b,:));
pth = max(pw(:))*1E-20;
if any(w =='p')
    y = log(max(m*pw,pth));
else
    ath = sqrt(pth);
    y = log(max(m*abs(f(a:b,:)),ath));
end
c = rdct(y).';
nf = size(c,1);
nc = nc+1;
if p>nc
    c(:,nc+1:end)=[];
elseif p<nc
    c=[c zeros(nf,nc-p)];
end
if ~any(w =='0') || flc > 0
    c(:,1) = [];
    nc = nc-1;
end
if any(w =='E')
    c = [log(max(sum(pw),pth)).' c];
    nc = nc+1;
end

% calculate derivative

if any(w =='D')
    vf = (4:-1:-4)/60;
    af = (1:-1:-1)/2;
    ww = ones(5,1);
    cx = [c(ww,:); c; c(nf*ww,:)];
    vx = reshape(filter(vf,1,cx(:)),nf+10,nc);
    vx(1:8,:) = [];
    ax=reshape(filter(af,1,vx(:)),nf+2,nc);
    ax(1:2,:) = [];
    vx([1 nf+2],:) = [];
    if any(w == 'd')
        c = [c vx ax];
    else
        c = [c ax];
    end
elseif any(w == 'd')
    vf = (4:-1:-4)/60;
    ww = ones(4,1);
    cx = [c(ww,:); c; c(nf*ww,:)];
    vx = reshape(filter(vf,1,cx(:)),nf+8,nc);
    vx(1:8,:) = [];
    c = [c vx];
end

if nargout<1
    [nf,nc] = size(c);
    t = ((0:nf-1)*inc+(n-1)/2)/fs;
    ci = (1:nc)-any(w =='0')-any(w=='E');
    imh = imagesc(t,ci,c.');
    axis('xy');
    xlabel('Time (s)');
    ylabel('Mel-cepstrum coefficient');
    map = (0:63)'/63;
    colormap([map map map]);
    colorbar;
end
[nf,nc] = size(c);
t = ((0:nf-1)*inc+(n-1)/2)/fs;
ci = (1:nc)-any(w=='0')-any(w=='E');
% % feature normalization by feature warping
% c = fea_warping(c', 301)';
% imagesc(t,ci,c.');
% axis('xy');
% xlabel('Time (secs)');
% ylabel('Mel-cepstrum coefficient');
% hcb = colorbar;
% ylabel(hcb, 'Normalized feature')

function Fea = fea_warping(fea, win)
% performs feature warping on feature streams over a sliding window
%
% Inputs:
%   - fea     : input ndim x nobs feature matrix, where nobs is the
%				number of frames and ndim is the feature dimension
%   - win     : length of the sliding window (should be an odd number)
%
% Outputs:
%   - Fea     : output ndim x nobs normalized feature matrix.
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

if ( nargin == 1),
    win = 301;
end

if ( mod(win, 2) == 0 ),
    fprintf(1, 'Error: Window length should be an odd number!\n');
    return;
end

fea = fea';
[nobs, ndim]  = size(fea);

if ( nobs < win ),
    if ~mod(nobs, 2)
        nobs = nobs + 1;
        fea = [fea; fea(end, :)];
    end
    win = nobs;
end

Fea = zeros(nobs, ndim);
[~, R] = sort(fea(1 : win, :));
[~, R] = sort(R);
arg = ( R(1 : ( win - 1 ) / 2, :) - 0.5 ) / win;
Fea(1 : ( win - 1 ) / 2, :) = norminv(arg, 0, 1);
for m = ( win - 1 ) / 2 + 1 : nobs - ( win - 1 ) / 2
    idx = m - ( win - 1 ) / 2 : m + ( win - 1 ) / 2;
    foo = fea(idx, :);
    R = sum(bsxfun(@lt, foo, foo(( win - 1 ) / 2 + 1, :))) + 1; % get the ranks
    arg = ( R - 0.5 ) / win;    % get the arguments
    Fea(m, :) = norminv(arg, 0, 1); % transform to normality
end
[~, R] = sort(fea(nobs - win + 1 : nobs, :));
[~, R] = sort(R);
arg = ( R( ( win + 1 ) / 2 + 1 : win, :) - 0.5 ) / win;
Fea(nobs - ( win - 1 ) / 2 + 1 : nobs, :) = norminv(arg, 0, 1);
Fea = Fea';


function f = enframe(x,win,inc)
%ENFRAME split signal up into (overlapping) frames: one per row. F=(X,WIN,INC)
nx=length(x);
nwin=length(win);
if (nwin == 1)
    len = win;
else
    len = nwin;
end
if (nargin < 3)
    inc = len;
end
nf = fix((nx-len+inc)/inc);
f=zeros(nf,len);
indf= inc*(0:(nf-1)).';
inds = (1:len);
f(:) = x(indf(:,ones(1,len))+inds(ones(nf,1),:));
if (nwin > 1)
    w = win(:)';
    f = f .* w(ones(nf,1),:);
end

function y = rfft(x,n,d)
s=size(x);
if prod(s)==1
    y = x;
else
    if nargin <3
        d=find(s>1);
        d=d(1);
        if nargin<2
            n=s(d);
        end
    end
    if isempty(n)
        n=s(d);
    end
    y=fft(x,n,d);
    y=reshape(y,prod(s(1:d-1)),n,prod(s(d+1:end)));
    s(d)=1+fix(n/2);
    y(:,s(d)+1:end,:)=[];
    y=reshape(y,s);
end

function y=rdct(x,n,a,b)
%RDCT     Discrete cosine transform of real data Y=(X,N,A,B)


fl=size(x,1)==1;
if fl x=x(:); end
[m,k]=size(x);
if nargin<4 b=1;
    if nargin<3 a=sqrt(2*m);
        if nargin<2 n=m;
        end
    end
end
if n>m x=[x; zeros(n-m,k)];
elseif n<m x(n+1:m,:)=[];
end

x=[x(1:2:n,:); x(2*fix(n/2):-2:2,:)];
z=[sqrt(2) 2*exp((-0.5i*pi/n)*(1:n-1))].';
y=real(fft(x).*z(:,ones(1,k)))/a;
y(1,:)=y(1,:)*b;
if fl y=y.'; end


% function [x,mn,mx] = melbankm(p,n,fs,fl,fh,w)
% %MELBANKM determine matrix for a mel-spaced filterbank [X,MN,MX]=(P,N,FS,FL,FH,W)
% w = 'tz';
% fh = 0.5;
% fl = 0;
%
% f0 = 700/fs;
% fn2 = floor(n/2);
% lr = log((f0+fh)/(f0+fl))/(p+1);
% % convert to fft bin numbers with 0 for DC term
% bl = n*((f0+fl)*exp([0 1 p p+1]*lr)-f0);
% b2 = ceil(bl(2));
% b3 = floor(bl(3));
%
% b1 = floor(bl(1))+1;
% b4 = min(fn2,ceil(bl(4)))-1;
% pf = log((f0+(b1:b4)/n)/(f0+fl))/lr;
% fp = floor(pf);
% pm = pf-fp;
% k2 = b2-b1+1;
% k3 = b3-b1+1;
% k4 = b4-b1+1;
% r = [fp(k2:k4) 1+fp(1:k3)];
% c = [k2:k4 1:k3];
% v = 2*[1-pm(k2:k4) pm(1:k3)];
% mn = b1+1;
% mx = b4+1;
%
% if nargout > 1
%     x=sparse(r,c,v);
% else
%     x=sparse(r,c+mn-1,v,p,1+fn2);
% end

function [x,mc,mn,mx]=melbankm(p,n,fs,fl,fh,w)
%MELBANKM determine matrix for a mel/erb/bark-spaced filterbank [X,MN,MX]=(P,N,FS,FL,FH,W)
%
% Inputs:
%       p   number of filters in filterbank or the filter spacing in k-mel/bark/erb [ceil(4.6*log10(fs))]
%		n   length of fft
%		fs  sample rate in Hz
%		fl  low end of the lowest filter as a fraction of fs [default = 0]
%		fh  high end of highest filter as a fraction of fs [default = 0.5]
%		w   any sensible combination of the following:
%             'b' = bark scale instead of mel
%             'e' = erb-rate scale
%             'l' = log10 Hz frequency scale
%             'f' = linear frequency scale
%
%             'c' = fl/fh specify centre of low and high filters
%             'h' = fl/fh are in Hz instead of fractions of fs
%             'H' = fl/fh are in mel/erb/bark/log10
%
%		      't' = triangular shaped filters in mel/erb/bark domain (default)
%		      'n' = hanning shaped filters in mel/erb/bark domain
%		      'm' = hamming shaped filters in mel/erb/bark domain
%
%		      'z' = highest and lowest filters taper down to zero [default]
%		      'y' = lowest filter remains at 1 down to 0 frequency and
%			        highest filter remains at 1 up to nyquist freqency
%
%             'u' = scale filters to sum to unity
%
%             's' = single-sided: do not double filters to account for negative frequencies
%
%             'g' = plot idealized filters [default if no output arguments present]
%
% Note that the filter shape (triangular, hamming etc) is defined in the mel (or erb etc) domain.
% Some people instead define an asymmetric triangular filter in the frequency domain.
%
%		       If 'ty' or 'ny' is specified, the total power in the fft is preserved.
%
% Outputs:	x     a sparse matrix containing the filterbank amplitudes
%		          If the mn and mx outputs are given then size(x)=[p,mx-mn+1]
%                 otherwise size(x)=[p,1+floor(n/2)]
%                 Note that the peak filter values equal 2 to account for the power
%                 in the negative FFT frequencies.
%           mc    the filterbank centre frequencies in mel/erb/bark
%		    mn    the lowest fft bin with a non-zero coefficient
%		    mx    the highest fft bin with a non-zero coefficient
%                 Note: you must specify both or neither of mn and mx.
%
% Examples of use:
%
% (a) Calcuate the Mel-frequency Cepstral Coefficients
%
%       f=rfft(s);			        % rfft() returns only 1+floor(n/2) coefficients
%		x=melbankm(p,n,fs);	        % n is the fft length, p is the number of filters wanted
%		z=log(x*abs(f).^2);         % multiply x by the power spectrum
%		c=dct(z);                   % take the DCT
%
% (b) Calcuate the Mel-frequency Cepstral Coefficients efficiently
%
%       f=fft(s);                        % n is the fft length, p is the number of filters wanted
%       [x,mc,na,nb]=melbankm(p,n,fs);   % na:nb gives the fft bins that are needed
%       z=log(x*(f(na:nb)).*conj(f(na:nb)));
%
% (c) Plot the calculated filterbanks
%
%      plot((0:floor(n/2))*fs/n,melbankm(p,n,fs)')   % fs=sample frequency
%
% (d) Plot the idealized filterbanks (without output sampling)
%
%      melbankm(p,n,fs);
%
% References:
%
% [1] S. S. Stevens, J. Volkman, and E. B. Newman. A scale for the measurement
%     of the psychological magnitude of pitch. J. Acoust Soc Amer, 8: 185?19, 1937.
% [2] S. Davis and P. Mermelstein. Comparison of parametric representations for
%     monosyllabic word recognition in continuously spoken sentences.
%     IEEE Trans Acoustics Speech and Signal Processing, 28 (4): 357?366, Aug. 1980.


%      Copyright (C) Mike Brookes 1997-2009
%      Version: $Id: melbankm.m 713 2011-10-16 14:45:43Z dmb $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   http://www.gnu.org/copyleft/gpl.html or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Note "FFT bin_0" assumes DC = bin 0 whereas "FFT bin_1" means DC = bin 1

if nargin < 6
    w='tz'; % default options
    if nargin < 5
        fh=0.5; % max freq is the nyquist
        if nargin < 4
            fl=0; % min freq is DC
        end
    end
end
sfact=2-any(w=='s');   % 1 if single sided else 2
wr=' ';   % default warping is mel
for i=1:length(w)
    if any(w(i)=='lebf');
        wr=w(i);
    end
end
if any(w=='h') || any(w=='H')
    mflh=[fl fh];
else
    mflh=[fl fh]*fs;
end
if ~any(w=='H')
    switch wr
        case 'f'       % no transformation
        case 'l'
            if fl<=0
                error('Low frequency limit must be >0 for l option');
            end
            mflh=log10(mflh);       % convert frequency limits into log10 Hz
        case 'e'
            mflh=frq2erb(mflh);       % convert frequency limits into erb-rate
        case 'b'
            mflh=frq2bark(mflh);       % convert frequency limits into bark
        otherwise
            mflh=frq2mel(mflh);       % convert frequency limits into mel
    end
end
melrng=mflh*(-1:2:1)';          % mel range
fn2=floor(n/2);     % bin index of highest positive frequency (Nyquist if n is even)
if isempty(p)
    p=ceil(4.6*log10(fs));         % default number of filters
end
if any(w=='c')              % c option: specify fiter centres not edges
    if p<1
        p=round(melrng/(p*1000))+1;
    end
    melinc=melrng/(p-1);
    mflh=mflh+(-1:2:1)*melinc;
else
    if p<1
        p=round(melrng/(p*1000))-1;
    end
    melinc=melrng/(p+1);
end

%
% Calculate the FFT bins corresponding to [filter#1-low filter#1-mid filter#p-mid filter#p-high]
%
switch wr
    case 'f'
        blim=(mflh(1)+[0 1 p p+1]*melinc)*n/fs;
    case 'l'
        blim=10.^(mflh(1)+[0 1 p p+1]*melinc)*n/fs;
    case 'e'
        blim=erb2frq(mflh(1)+[0 1 p p+1]*melinc)*n/fs;
    case 'b'
        blim=bark2frq(mflh(1)+[0 1 p p+1]*melinc)*n/fs;
    otherwise
        blim=mel2frq(mflh(1)+[0 1 p p+1]*melinc)*n/fs;
end
mc=mflh(1)+(1:p)*melinc;    % mel centre frequencies
b1=floor(blim(1))+1;            % lowest FFT bin_0 required might be negative)
b4=min(fn2,ceil(blim(4))-1);    % highest FFT bin_0 required
%
% now map all the useful FFT bins_0 to filter1 centres
%
switch wr
    case 'f'
        pf=((b1:b4)*fs/n-mflh(1))/melinc;
    case 'l'
        pf=(log10((b1:b4)*fs/n)-mflh(1))/melinc;
    case 'e'
        pf=(frq2erb((b1:b4)*fs/n)-mflh(1))/melinc;
    case 'b'
        pf=(frq2bark((b1:b4)*fs/n)-mflh(1))/melinc;
    otherwise
        pf=(frq2mel((b1:b4)*fs/n)-mflh(1))/melinc;
end
%
%  remove any incorrect entries in pf due to rounding errors
%
if pf(1)<0
    pf(1)=[];
    b1=b1+1;
end
if pf(end)>=p+1
    pf(end)=[];
    b4=b4-1;
end
fp=floor(pf);                  % FFT bin_0 i contributes to filters_1 fp(1+i-b1)+[0 1]
pm=pf-fp;                       % multiplier for upper filter
k2=find(fp>0,1);   % FFT bin_1 k2+b1 is the first to contribute to both upper and lower filters
k3=find(fp<p,1,'last');  % FFT bin_1 k3+b1 is the last to contribute to both upper and lower filters
k4=numel(fp); % FFT bin_1 k4+b1 is the last to contribute to any filters
if isempty(k2)
    k2=k4+1;
end
if isempty(k3)
    k3=0;
end
if any(w=='y')          % preserve power in FFT
    mn=1; % lowest fft bin required (1 = DC)
    mx=fn2+1; % highest fft bin required (1 = DC)
    r=[ones(1,k2+b1-1) 1+fp(k2:k3) fp(k2:k3) repmat(p,1,fn2-k3-b1+1)]; % filter number_1
    c=[1:k2+b1-1 k2+b1:k3+b1 k2+b1:k3+b1 k3+b1+1:fn2+1]; % FFT bin1
    v=[ones(1,k2+b1-1) pm(k2:k3) 1-pm(k2:k3) ones(1,fn2-k3-b1+1)];
else
    r=[1+fp(1:k3) fp(k2:k4)]; % filter number_1
    c=[1:k3 k2:k4]; % FFT bin_1 - b1
    v=[pm(1:k3) 1-pm(k2:k4)];
    mn=b1+1; % lowest fft bin_1
    mx=b4+1;  % highest fft bin_1
end
if b1<0
    c=abs(c+b1-1)-b1+1;     % convert negative frequencies into positive
end
% end
if any(w=='n')
    v=0.5-0.5*cos(v*pi);      % convert triangles to Hanning
elseif any(w=='m')
    v=0.5-0.46/1.08*cos(v*pi);  % convert triangles to Hamming
end
if sfact==2  % double all except the DC and Nyquist (if any) terms
    msk=(c+mn>2) & (c+mn<n-fn2+2);  % there is no Nyquist term if n is odd
    v(msk)=2*v(msk);
end
%
% sort out the output argument options
%
if nargout > 2
    x=sparse(r,c,v);
    if nargout == 3     % if exactly three output arguments, then
        mc=mn;          % delete mc output for legacy code compatibility
        mn=mx;
    end
else
    x=sparse(r,c+mn-1,v,p,1+fn2);
end
if any(w=='u')
    sx=sum(x,2);
    x=x./repmat(sx+(sx==0),1,size(x,2));
end
%
% plot results if no output arguments or g option given
%
if ~nargout || any(w=='g') % plot idealized filters
    ng=201;     % 201 points
    me=mflh(1)+(0:p+1)'*melinc;
    switch wr
        case 'f'
            fe=me; % defining frequencies
            xg=repmat(linspace(0,1,ng),p,1).*repmat(me(3:end)-me(1:end-2),1,ng)+repmat(me(1:end-2),1,ng);
        case 'l'
            fe=10.^me; % defining frequencies
            xg=10.^(repmat(linspace(0,1,ng),p,1).*repmat(me(3:end)-me(1:end-2),1,ng)+repmat(me(1:end-2),1,ng));
        case 'e'
            fe=erb2frq(me); % defining frequencies
            xg=erb2frq(repmat(linspace(0,1,ng),p,1).*repmat(me(3:end)-me(1:end-2),1,ng)+repmat(me(1:end-2),1,ng));
        case 'b'
            fe=bark2frq(me); % defining frequencies
            xg=bark2frq(repmat(linspace(0,1,ng),p,1).*repmat(me(3:end)-me(1:end-2),1,ng)+repmat(me(1:end-2),1,ng));
        otherwise
            fe=mel2frq(me); % defining frequencies
            xg=mel2frq(repmat(linspace(0,1,ng),p,1).*repmat(me(3:end)-me(1:end-2),1,ng)+repmat(me(1:end-2),1,ng));
    end
    
    v=1-abs(linspace(-1,1,ng));
    if any(w=='n')
        v=0.5-0.5*cos(v*pi);      % convert triangles to Hanning
    elseif any(w=='m')
        v=0.5-0.46/1.08*cos(v*pi);  % convert triangles to Hamming
    end
    v=v*sfact;  % multiply by 2 if double sided
    v=repmat(v,p,1);
    if any(w=='y')  % extend first and last filters
        v(1,xg(1,:)<fe(2))=sfact;
        v(end,xg(end,:)>fe(p+1))=sfact;
    end
    if any(w=='u') % scale to unity sum
        dx=(xg(:,3:end)-xg(:,1:end-2))/2;
        dx=dx(:,[1 1:ng-2 ng-2]);
        vs=sum(v.*dx,2);
        v=v./repmat(vs+(vs==0),1,ng)*fs/n;
    end
    plot(xg',v','b');
    set(gca,'xlim',[fe(1) fe(end)]);
    xlabel(['Frequency (' xticksi 'Hz)']);
end

function [mel,mr] = frq2mel(frq)
%FRQ2ERB  Convert Hertz to Mel frequency scale MEL=(FRQ)
%	[mel,mr] = frq2mel(frq) converts a vector of frequencies (in Hz)
%	to the corresponding values on the Mel scale which corresponds
%	to the perceived pitch of a tone.
%   mr gives the corresponding gradients in Hz/mel.

%	The relationship between mel and frq is given by [1]:
%
%	m = ln(1 + f/700) * 1000 / ln(1+1000/700)
%
%  	This means that m(1000) = 1000
%
%	References:
%
%     [1] J. Makhoul and L. Cosell. "Lpcw: An lpc vocoder with
%         linear predictive spectral warping", In Proc IEEE Intl
%         Conf Acoustics, Speech and Signal Processing, volume 1,
%         pages 466?469, 1976. doi: 10.1109/ICASSP.1976.1170013.
%	  [2] S. S. Stevens & J. Volkman "The relation of pitch to
%		  frequency", American J of Psychology, V 53, p329 1940
%	  [3] C. G. M. Fant, "Acoustic description & classification
%		  of phonetic units", Ericsson Tchnics, No 1 1959
%		  (reprinted in "Speech Sounds & Features", MIT Press 1973)
%	  [4] S. B. Davis & P. Mermelstein, "Comparison of parametric
%		  representations for monosyllabic word recognition in
%		  continuously spoken sentences", IEEE ASSP, V 28,
%		  pp 357-366 Aug 1980
%	  [5] J. R. Deller Jr, J. G. Proakis, J. H. L. Hansen,
%		  "Discrete-Time Processing of Speech Signals", p380,
%		  Macmillan 1993

%      Copyright (C) Mike Brookes 1998
%      Version: $Id: frq2mel.m 1874 2012-05-25 15:41:53Z dmb $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   http://www.gnu.org/copyleft/gpl.html or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
persistent k
if isempty(k)
    k=1000/log(1+1000/700); %  1127.01048
end
af=abs(frq);
mel = sign(frq).*log(1+af/700)*k;
mr=(700+af)/k;
if ~nargout
    plot(frq,mel,'-x');
    xlabel(['Frequency (' xticksi 'Hz)']);
    ylabel(['Frequency (' yticksi 'Mel)']);
end

function [frq,mr] = mel2frq(mel)
%MEL2FRQ  Convert Mel frequency scale to Hertz FRQ=(MEL)
%	frq = mel2frq(mel) converts a vector of Mel frequencies
%	to the corresponding real frequencies.
%   mr gives the corresponding gradients in Hz/mel.
%	The Mel scale corresponds to the perceived pitch of a tone

%	The relationship between mel and frq is given by [1]:
%
%	m = ln(1 + f/700) * 1000 / ln(1+1000/700)
%
%  	This means that m(1000) = 1000
%
%	References:
%
%     [1] J. Makhoul and L. Cosell. "Lpcw: An lpc vocoder with
%         linear predictive spectral warping", In Proc IEEE Intl
%         Conf Acoustics, Speech and Signal Processing, volume 1,
%         pages 466?469, 1976. doi: 10.1109/ICASSP.1976.1170013.
%	  [2] S. S. Stevens & J. Volkman "The relation of pitch to
%		  frequency", American J of Psychology, V 53, p329 1940
%	  [3] C. G. M. Fant, "Acoustic description & classification
%		  of phonetic units", Ericsson Tchnics, No 1 1959
%		  (reprinted in "Speech Sounds & Features", MIT Press 1973)
%	  [4] S. B. Davis & P. Mermelstein, "Comparison of parametric
%		  representations for monosyllabic word recognition in
%		  continuously spoken sentences", IEEE ASSP, V 28,
%		  pp 357-366 Aug 1980
%	  [5] J. R. Deller Jr, J. G. Proakis, J. H. L. Hansen,
%		  "Discrete-Time Processing of Speech Signals", p380,
%		  Macmillan 1993

%      Copyright (C) Mike Brookes 1998
%      Version: $Id: mel2frq.m 1874 2012-05-25 15:41:53Z dmb $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   http://www.gnu.org/copyleft/gpl.html or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
persistent k
if isempty(k)
    k=1000/log(1+1000/700); % 1127.01048
end
frq=700*sign(mel).*(exp(abs(mel)/k)-1);
mr=(700+abs(frq))/k;
if ~nargout
    plot(mel,frq,'-x');
    xlabel(['Frequency (' xticksi 'Mel)']);
    ylabel(['Frequency (' yticksi 'Hz)']);
end

% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
% delete(hObject);
selection = questdlg('Close Speaker Verification?',...
    'Close Request Function',...
    'Yes','No','Yes');
switch selection
    case 'Yes'
        %         if ~isempty(handles)
        %             fgen = handles.setupdata.fgen;
        %             if ~isempty(fgen)
        %                 %             fprintf(fgen,'OUTPUT1 OFF'); %Disable Output for channel 1
        %                 fclose(fgen);
        %                 handles.setupdata.fgen = [];
        %             end
        %         end
        delete(hObject);
    case 'No'
        return
end


% --- Executes on selection change in pm_DataBase.
function pm_DataBase_Callback(hObject, eventdata, handles)
% hObject    handle to pm_DataBase (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pm_DataBase contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pm_DataBase

contents = cellstr(get(handles.pm_DataBase,'String'));
ncurDatabase = get(handles.pm_DataBase,'value');
dataBasefileName = [contents{ncurDatabase} '.mat'];
if ~exist( dataBasefileName,'file')
    warndlg('Default Feature file not found!','Error');
    handles.database.no_of_fe = 0;
else
    handles.database = load(dataBasefileName);
    handles.database.no_of_fe = length(handles.database.gmm_models);
end
if isempty(handles.database.no_of_fe)
    handles.database.no_of_fe = 0;
end
if (handles.database.no_of_fe==0)
    handles.database.name = char('');
end

if ~isempty(handles.database.gmm_models)
    speakerName = cell(1,length(handles.database.gmm_models));
    for i = 1:length(handles.database.gmm_models)
        speakerName{i} = handles.database.gmm_models{1, i}.name;
    end
end
set(handles.pm_SpeakerName,'String',...
    speakerName)
handles.speakerName = speakerName;
set(handles.text_DataBaseNumber,'String',handles.database.no_of_fe);
handles.database.dataBasefileName = dataBasefileName;
guidata(hObject,handles)


% --- Executes during object creation, after setting all properties.
function pm_DataBase_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pm_DataBase (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on button press in bt_Recognition.
function bt_Recognition_Callback(hObject, eventdata, handles)
% hObject    handle to bt_Recognition (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isempty(handles.database.ubm) || isempty(handles.database.gmm_models)
    errordlg('Oh dear, without UBM and Speakermodels, I can not do this!')
    return
end
ubm = handles.database.ubm;
gmm_models = handles.database.gmm_models;
norders = handles.database.norders;
fl = handles.database.fl;
fh = handles.database.fh;
nSpeaker = length(gmm_models);

Fs = handles.setupdata.Fs;
curSignal = handles.setupdata.curSignal;

switch norders
    case 3
        curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'MDd0');
    case 2
        curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'MD0');
    case 1
        curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'M0');
end
curNormFeature = feanormalize(curmfccFeature,'w',Fs); % 'w' for warping or 'c' for cmvn
likelihoodindex = zeros(1,nSpeaker);
for ispeaker = 1:nSpeaker
    gmm = gmm_models{1,ispeaker};
    ubm_llk = compute_llk(curNormFeature', ubm.mu, ubm.sigma, ubm.w(:));
    gmm_llk = compute_llk(curNormFeature', gmm.mu, gmm.sigma, gmm.w(:));
    likelihoodindex(ispeaker) = mean(gmm_llk - ubm_llk);
end
% matchindex = mean(likelihoodindex,2);
[dematchindex,imatch] = sort(likelihoodindex,'descend');
% qualityindex = abs(dematchindex(1) -mean(dematchindex));
% if qualityindex > 1
%     set(handles.text_quality,'string','PERFECT','visible','on');
set(handles.text_result,'string',gmm_models{1,imatch(1)}.name,'visible','on');
set(handles.text_likelihood,'string',dematchindex(1),'visible','on');
% else
%     set(handles.text_quality,'string','POOR','visible','on');
%     set(handles.text_result,'string',name(imatch(1),:),'visible','on');
%     set(handles.text_likelihood,'string',dematchindex(1),'visible','on');
% end
axes(handles.axes_likelihood)
bar(likelihoodindex)
xlabel('Speaker#')
ylabel('Likelihoods')




% --- Executes on button press in bt_Record.
function bt_Record_Callback(hObject, eventdata, handles)
% hObject    handle to bt_Record (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

list = {'Ultrasonic micphone, 192kHz','MEMS micphone, 48kHz','Laptop, 48kHz',...
    };
[recordindx,~] = listdlg('ListString',list,'SelectionMode','single','PromptString','Select a Recorder' );
def = {'2'};
prompt = {'Enter record time (secs):'};
dlg_title = 'Input for record ';
num_lines = 1;
answer = inputdlg(prompt,dlg_title,num_lines,def);

if ~isempty(answer)
    recordTime = str2num(answer{1});
else
    return
end
if ~isempty(recordindx)
    %     recordTime = 5;
    set(handles.bt_Record,'String','Recording')
    switch recordindx
        case 1 % 'Ultrasonic micphone, 200kHz'
            Fs = 192e3;
            flagultramic = 1;
            [myRecording,myRecording1] = dataacquisition(Fs,recordTime,flagultramic);
        case 2 % 'MEMS micphone, 48kHz'
            Fs = 48e3;
            flagultramic = 0;
            myRecording = dataacquisition(Fs,recordTime,flagultramic);
        case 3 % 'Laptop, 48kHz'
            Fs = 48e3;
            flagultramic = 0;
            % Record your voice for 5 seconds.
            nBits = 24;
            nChannels = 1;
            recObj = audiorecorder(Fs,nBits,nChannels);
            %             disp('Start speaking.')
            recordblocking(recObj, recordTime);
            %             disp('End of Recording.');
            %             % Play back the recording.
            %             play(recObj);
            % Store data in double-precision array.
            myRecording = getaudiodata(recObj);
    end
else
    return
end
% finish record.
set(handles.bt_Record,'String','Record')
% Plot the waveform.
myRecording = myRecording-mean(myRecording); % remove DC
myRecording = myRecording./max(abs(myRecording));
axes(handles.axes_Wave);
cla(gca)
plotWave(handles.axes_Wave,Fs,myRecording)
title(handles.axes_Wave,'new recording');


% Plot second record 2020,1/22
myRecording1 = myRecording1-mean(myRecording1); % remove DC
myRecording1 = myRecording1./max(abs(myRecording1));
axes(handles.axes23);
cla(gca)
plotWave(handles.axes23,Fs,myRecording1)
title(handles.axes23,'new new recording');

% Plot the PSD
axes(handles.axes_PSD);
cla(gca)
plotPSD(handles.axes_PSD,Fs,myRecording)
% cla figure
axes(handles.axes_Spectrogram);
cla(gca)
axes(handles.axes_Feature);
cla(gca)

handles.setupdata.Fs = Fs;
handles.setupdata.curSignal = myRecording;
handles.setupdata.curSignal1 = myRecording1;
handles.setupdata.flagultramic = flagultramic;
handles.setupdata.curFileName = [];
set(handles.text_result,'string','','visible','off');
set(handles.text_quality,'string','','visible','off');
set(handles.text_likelihood,'string','','visible','off');
guidata(hObject, handles);

function plotPSD(axesplot,Fs,curOrignal)
nSignal = length(curOrignal);
[pxx,f] = periodogram(curOrignal,hamming(nSignal),nSignal,Fs);
plot(axesplot,f/1e3,log(pxx),'LineWidth',1.5);
xlabel('Freq (kHz)')
ylabel('Power/Freq (dB/Hz)')
title('periodogram PSD Estimation')
xlim([0,Fs/2/1e3])


function plotWave(axesplot,Fs,curOrignal)
% Plot the waveform.
tvect = (0:(length(curOrignal)-1))/Fs;
plot(axesplot,tvect,curOrignal,'LineWidth',1.5);
xlabel('Time (sec)')
ylabel('Amplitude')
% axis tight

function [signal,signal1] = dataacquisition(Fs,recordTime,flagRange)
%% Create Data Acquisition Session
% Create a session for the specified vendor.
s = daq.createSession('ni');
%% Set Session Properties
% Set properties that are not using default values.
s.Rate = Fs;
s.DurationInSeconds = recordTime;
%% Add Channels to Session
% Add channels and set channel properties, if any.
channel1 = addAnalogInputChannel(s,'Dev1','ai0','Voltage');
channel1.TerminalConfig = 'SingleEnded';
if flagRange == 1 % ultramic
    channel1.Range = [-1.000000 1.000000];
else % Acoustic mic
    channel1.Range = [-10.000000 10.000000];
end

%% Add the other channel
channel2 = addAnalogInputChannel(s, 'Dev1', 'ai1', 'Voltage');
channel2.TerminalConfig = 'SingleEnded';
if flagRange == 1 % ultramic
    channel2.Range = [-1.000000 1.000000];
else % Acoustic mic
    channel2.Range = [-10.000000 10.000000];
end
%% Acquire Data
% Start the session in foreground.
%   [data, timestamps, starttime] = startForeground(s);
[data, ~, ~] = startForeground(s);
%% Log Data
% Convert the acquired data and timestamps to a timetable in a workspace variable.
signal = data(:,1);
signal1 = data(:,2); %second channel signal
% Clear the session and channels, if any.
clear s channel1 channel2


% --- Executes on button press in bt_Play.
function bt_Play_Callback(hObject, eventdata, handles)
% hObject    handle to bt_Play (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Fs = handles.setupdata.Fs;
curSignal = handles.setupdata.curSignal;
audiowavplay(curSignal,Fs)

% --- Executes on button press in bt_Selectionplay.
function bt_Selectionplay_Callback(hObject, eventdata, handles)
% hObject    handle to bt_Selectionplay (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

axes(handles.axes_Wave)
cla(gca)
Fs = handles.setupdata.Fs;
curSignal = handles.setupdata.curSignal;
% Allow the user to draw a rectangle on the area
% they would like to zoom in on
RECT = getrect;

xmin = RECT(1);
xmax = RECT(1) + RECT(3);
ymin = RECT(2);
ymax = RECT(2) + RECT(4);

% Set maximum zoom limits to the data edges
xaxis_limits = get(handles.axes_Wave,'XLim');
yaxis_limits = get(handles.axes_Wave,'YLim');
yaxis_limits(2);
xaxis_limits(2);

if xmin < xaxis_limits(1)
    xmin = xaxis_limits(1);
end

if xmax > xaxis_limits(2)
    xmax = xaxis_limits(2);
    
end

if ymin < yaxis_limits(1)
    ymin = yaxis_limits(1);
end

if ymax > yaxis_limits(2)
    ymax = yaxis_limits(2);
    yaxis_limits(2);
end

% if the choosen zoom range is acceptable...
if ~((ymin > ymax) || (xmin > xmax))
    
    % zoom in on the frequency data by adjusting the xaxis
    % limits to be the same as those of the time data
    % define the zoomed in data (for playback purposes)
    imin = round(xmin*Fs)+1;
    imax = round(xmax*Fs)+1;
end
playSignal = curSignal(imin:imax);
audiowavplay(playSignal,Fs)

function audiowavplay(playSignal,Fs)
if ~isempty(playSignal)
    if Fs > 48e3 && mod(Fs,48e3)==0
        y = downsample(playSignal,Fs/48e3);
        wavplay = audioplayer(y, 48e3);
    else
        wavplay = audioplayer(playSignal, Fs);
    end
    playblocking(wavplay);
end

% --- Executes on button press in bt_selectiondelete.
function bt_selectiondelete_Callback(hObject, eventdata, handles)
% hObject    handle to bt_selectiondelete (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

axes(handles.axes_Wave)
Fs = handles.setupdata.Fs;
curSignal = handles.setupdata.curSignal;

% Allow the user to draw a rectangle on the area
% they would like to zoom in on
RECT = getrect;

xmin = RECT(1);
xmax = RECT(1) + RECT(3);
ymin = RECT(2);
ymax = RECT(2) + RECT(4);

% Set maximum zoom limits to the data edges
xaxis_limits = get(handles.axes_Wave,'XLim');
yaxis_limits = get(handles.axes_Wave,'YLim');
yaxis_limits(2);
xaxis_limits(2);

if xmin < xaxis_limits(1)
    xmin = xaxis_limits(1);
end

if xmax > xaxis_limits(2)
    xmax = xaxis_limits(2);
    
end

if ymin < yaxis_limits(1)
    ymin = yaxis_limits(1);
end

if ymax > yaxis_limits(2)
    ymax = yaxis_limits(2);
    yaxis_limits(2);
end

% if the choosen zoom range is acceptable...
if ~((ymin > ymax) || (xmin > xmax))||(size(curSignal)<10)
    
    % zoom in on the frequency data by adjusting the xaxis
    % limits to be the same as those of the time data
    % define the zoomed in data (for playback purposes)
    imin = round(xmin*Fs) +1;
    imax = round(xmax*Fs) ;
    imax =  min(length(curSignal),imax);
    button = questdlg('DO YOU WANT TO DELETE','Delete');
    switch button
        case {'Yes'}
            curSignal(imin:imax)=[];
            axes(handles.axes_Wave);
            cla(gca)
            plotWave(handles.axes_Wave,Fs,curSignal)
            % Plot the PSD
            axes(handles.axes_PSD);
            cla(gca)
            plotPSD(handles.axes_PSD,Fs,curSignal)
    end
end
handles.setupdata.curSignal = curSignal;
guidata(hObject, handles);

% --- Executes on button press in bt_Load.
function bt_Load_Callback(hObject, eventdata, handles)
% hObject    handle to bt_Load (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[filename, pathname] = uigetfile({'*.wav';...
    '*.mp3'},'select a wave file to load');
if filename == 0
    errordlg('ERROR! No file selected!');
    return;
end
% check file is selected

[signal,Fs] = audioread([pathname filename]);
signal = signal - mean(signal);
curSignal = signal/max(abs(signal));

% displays the time graph of the voice signal
axes(handles.axes_Wave);
cla(gca)
plotWave(handles.axes_Wave,Fs,curSignal)
title(handles.axes_Wave,filename);
% Plot the PSD 
axes(handles.axes_PSD);
cla(gca)
plotPSD(handles.axes_PSD,Fs,curSignal)
% cla
axes(handles.axes_Spectrogram);
cla(gca)
axes(handles.axes_Feature);
cla(gca)

handles.setupdata.Fs = Fs;
handles.setupdata.curSignal = curSignal;
handles.setupdata.curFileName = filename;
% handles.setupdata.flagultramic = flagultramic;
set(handles.text_result,'string','','visible','off');
set(handles.text_quality,'string','','visible','off');
set(handles.text_likelihood,'string','','visible','off');
guidata(hObject,handles)

% --- Executes on button press in bt_Save.
function bt_Save_Callback(hObject, eventdata, handles)
% hObject    handle to bt_Save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

Fs = handles.setupdata.Fs;
curSignal = handles.setupdata.curSignal;
[filename, pathname] = uiputfile('*.wav', 'Save Data to Wave File');
if filename ~= 0
    audiowrite([pathname filename],curSignal,Fs)
end
title(handles.axes_Wave,filename);
handles.setupdata.curFileName = filename;
guidata(hObject,handles)




% --- Executes on button press in bt_GMM.
function bt_GMM_Callback(hObject, eventdata, handles)
% hObject    handle to bt_GMM (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

Fs = handles.setupdata.Fs;
feature = handles.mfccFeature;
No_of_Gaussians = 12;
axes(handles.axes_Feature)
[feature,t,ci] = feanormalize(feature,'w',Fs); % 'w' for warping or 'c' for cmvn
imagesc(t,ci,feature.');
axis('xy');
xlabel('Time (secs)');
ylabel('Mel-cepstrum coefficient');
hcb = colorbar;
ylabel(hcb, 'Normalized feature')

[mu_train,sigma_train,c_train,LLH_train] = gmm_estimate(feature',No_of_Gaussians,20);
handles.traindata.mu_train = mu_train;
handles.traindata.sigma_train = sigma_train;
handles.traindata.c_train = c_train;
disp(LLH_train)
guidata(hObject,handles)


function [fea,t,ci] = feanormalize(fea,w,Fs)
if w=='w'
    % feature normalization by feature warping
    fea = fea_warping(fea', 1801)';
elseif w=='c'
    % feature normalization by cmvn
    fea = wcmvn(fea, 1801, true);
else
    return
end
[nf,nc] = size(fea);
n = pow2(floor(log2(0.03*Fs)));
inc = floor(n/2);
t = ((0:nf-1)*inc+(n-1)/2)/Fs;
ci = (1:nc)-1;


function Fea = wcmvn(fea, win, varnorm)
% performs cepstral mean and variance normalization over a sliding window
%
% Inputs:
%   - fea     : input ndim x nobs feature matrix, where nobs is the
%				number of frames and ndim is the feature dimension
%   - win     : length of the sliding window (should be an odd number)
%   - varnorm : binary switch (false|true), if true variance is normalized
%               as well
% Outputs:
%   - Fea     : output ndim x nobs normalized feature matrix.

% Example:
%         wcmvn(mfc(2,:), 301, true)
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center


if ( nargin < 3 ),
    varnorm = false;
end
if ( nargin == 1),
    win = 301;
end

if ( mod(win, 2) == 0 ),
    fprintf(1, 'Error: Window length should be an odd number!\n');
    return;
end

[ndim, nobs]  = size(fea);
if ( nobs < win ),
    if ~mod(nobs, 2)
        nobs = nobs+1;
        fea = [fea, fea(:, end)];
    end
    win = nobs;
end

epss = 1e-20;
Fea = zeros(ndim, nobs);
idx = 1 : ( win - 1 )/2;
Fea(:, idx) = bsxfun(@minus, fea(:, idx), mean(fea(:, 1 : win), 2));

if varnorm,
    Fea(:, idx) = bsxfun(@rdivide, Fea(:, idx), std(fea(:, 1 : win), [], 2) + epss);
    
    for m = ( win - 1 )/2 + 1 : nobs - ( win - 1 )/2
        idx = m - ( win - 1 )/2 : m + ( win - 1 )/2;
        Fea(:, m) = ( fea(:, m) - mean(fea(:, idx), 2) )./(std(fea(:, idx), [], 2) + epss);
    end
else
    for m = ( win - 1 )/2 + 1 : nobs - ( win - 1 )/2
        idx = m - ( win - 1 )/2 : m + ( win - 1 )/2;
        Fea(:, m) = ( fea(:, m) - mean(fea(:, idx), 2) );
    end
end

idx = (nobs - ( win - 1 )/2 + 1) : nobs;
Fea(:, idx) = bsxfun(@minus, fea(:, idx), mean(fea(:, nobs-win+1:nobs), 2));

if varnorm,
    Fea(:, idx) = bsxfun(@rdivide, Fea(:, idx), std(fea(:, nobs-win+1:nobs), [], 2) + epss);
end


function [mu,sigm,c,LLH] = gmm_estimate(X,M,iT,mu,sigm,c,Vm)
% [mu,sigma,c] = gmm_estimate(X,M,<iT,mu,sigm,c,Vm>)
%
% X   : the column by column data matrix (LxT)
% M   : number of gaussians
% iT  : number of iterations, by defaut 10
% mu  : initial means (LxM)
% sigm: initial diagonals for the diagonal covariance matrices (LxM)
% c   : initial weights (Mx1)
% Vm  : minimal variance factor, by defaut 4 ->minsig = var/(M�Vm?)



% *************************************************************
% GENERAL PARAMETERS
[L,T] = size(X);        % data length
varL = var(X')';    % variance for each row data;

min_diff_LLH = 0.001;   % convergence criteria

% DEFAULTS
if nargin <= 2
    iT = 10;  % number of iterations,
end
if nargin<=3
    mu = X(:,[fix((T-1).*rand(1,M))+1]);  % mu def: M rand vect.
    sigm = repmat(varL./(M.^2),[1,M]);  % sigm def: same variance
    c = ones(M,1)./M;   % c def: same weight
    Vm = 4;    % minimum variance factor
end

min_sigm = repmat(varL./(Vm.^2*M.^2),[1,M]);   % MINIMUM sigma!

% VARIABLES
% lgam_m = zeros(T,M);    % prob of each (X:,t) to belong to the kth mixture
% lB = zeros(T,1);        % log-likelihood
% lBM = zeros(T,M);       % log-likelihhod for separate mixtures


old_LLH = -9e99;        % initial log-likelihood

% START ITERATATIONS
for iter = 1:iT
    
    
    % ESTIMATION STEP ****************************************************
    [lBM,lB] = lmultigauss(X,mu,sigm,c);
    
    
    LLH = mean(lB);
    %set(handles.tx_msg,'String','Extracting MFCC Features');
    %disp(sprintf('log-likelihood :  %f',LLH));
    
    lgam_m = lBM-repmat(lB,[1,M]);  % logarithmic version
    gam_m = exp(lgam_m);            % linear version           -Equation(1)
    
    
    % MAXIMIZATION STEP **************************************************
    sgam_m = sum(gam_m);            % sum of gam_m for all X(:,t)
    
    
    % gaussian weights ************************************
    new_c = mean(gam_m)';      %                                -Equation(4)
    
    % means    ********************************************
    % (convert gam_m and X to (L,M,T) and .* and then sum over T)
    mu_numerator = sum(permute(repmat(gam_m,[1,1,L]),[3,2,1]).*...
        permute(repmat(X,[1,1,M]),[1,3,2]),3);
    % convert  sgam_m(1,M,N) -> (L,M,N) and then ./
    new_mu = mu_numerator./repmat(sgam_m,[L,1]);              % -Equation(2)
    
    % variances *******************************************
    sig_numerator = sum(permute(repmat(gam_m,[1,1,L]),[3,2,1]).*...
        permute(repmat(X.*X,[1,1,M]),[1,3,2]),3);
    
    new_sigm = sig_numerator./repmat(sgam_m,[L,1])-new_mu.^2; % -Equation(3)
    
    % the variance is limited to a minimum
    new_sigm = max(new_sigm,min_sigm);
    
    %*******
    % UPDATE
    
    if old_LLH >= LLH - min_diff_LLH
        
        break;
    else
        
        old_LLH = LLH;
        
        mu = new_mu;
        sigm = new_sigm;
        c = new_c;
        
    end
    
    %******************************************************************
end

function [YM,Y] = lmultigauss(x,mus,sigm,c)
% [lYM,lY] = lmultigauss(X,mu,sigm,c)
%
% computes multigaussian log-likelihood
%
% X   : (LxT) data (columnwise vectors)
% sigm: (LxM) variances vector  (diagonal of the covariance matrix)
% mu  : (LxM) means
% c   : (Mx1) the weights



[L,T] = size(x);
M = size(c,1);


% repeating, changing dimensions:
X = permute(repmat(x',[1,1,M]),[1,3,2]);      % (T,L) -> (T,M,L) one per mixture

Sigm = permute(repmat(sigm,[1,1,T]),[3,2,1]); % (L,M) -> (T,M,L)

Mu = permute(repmat(mus,[1,1,T]),[3,2,1]);     % (L,M) -> (T,M,L)


%Y = squeeze(exp( 0.5.*dot(X-Mu,(X-Mu)./Sigm))) % L dissapears: (L,T,M) -> (T,M)
lY = -0.5.*dot(X-Mu,(X-Mu)./Sigm,3);
% c,const -> (T,M) and then multiply by old Y
lcoi = log(2.*pi).*(L./2)+0.5.*sum(log(sigm),1); % c,const -> (T,M)
lcoef = repmat(log(c') - lcoi,[T,1]);


YM = lcoef + lY;            % ( T,M ) one mixture per column
Y = lsum(YM,2);                 % add mixtures

function lz=lsum(X,DIM)
% lz=lsum(X,DIM);
%
% lz=log(x(1)+x(2)+....x(n))  X(i)= log(x(i)) ;
%
% lsum(X)     sums along first dimension
% lsum(X,DIM) sums along dimension DIM


if nargin==1
    DIM=1;
end

s=size(X);

if DIM == 1
    % formula is:
    % lz=log(bigger)+log(1+sum(exp(log(others)-log(bigger))))
    
    % ************************************************************
    X=sort(X,1);   % just for find bigger in all dimensions
    lz=X(end,:,:,:,:)+...
        log(1+sum(exp(X(1:end-1,:,:,:,:)-...
        repmat(X(end,:,:,:,:),[size(X,1)-1,1,1,1,1])),1));
    % ************************************************************
else
    % we put DIM to first dimension and do the same as before
    X=permute(X,[ DIM, 1:DIM-1 , DIM+1:length(s)]);
    
    % ************************************************************
    X=sort(X,1);
    lz=X(end,:,:,:,:)+...
        log(1+sum(exp(X(1:end-1,:,:,:,:)-...
        repmat(X(end,:,:,:,:),[size(X,1)-1,1,1,1,1])),1));
    % *************************************************************
    
    lz=permute(lz,[2:DIM, 1, DIM+1:length(s)]);
    % we bring back dimesions
end


% --- Executes on button press in bt_Add2database.
function bt_Add2database_Callback(hObject, eventdata, handles)
% hObject    handle to bt_Add2database (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isempty(handles.database.ubm) || isempty(handles.database.gmm_models)
    errordlg('Oh dear, without UBM and Speakermodels, I can not do this!')
    return
end
curName = handles.setupdata.curFileName;
if isempty(curName)
    def = {''};
else
    def = {curName(1:(end-4))};
end
prompt = {'Enter Name:'};
dlg_title = 'Input for adding to database ';
num_lines = 1;
answer = inputdlg(prompt,dlg_title,num_lines,def);

if ~isempty(answer)
    per_name = answer{1};
else
    return
end
indcmp = zeros(1,length(handles.database.gmm_models));
for i = 1:length(handles.database.gmm_models)
    indcmp(i) = strcmp(per_name,handles.database.gmm_models{1,i}.name);
end
if any(indcmp>0)  % find if the new name already exists in data base
    errordlg('name Already exists');
    return
else
    selection = questdlg(strcat('Save the Voice with name as :',per_name),...
        'Close Request Function',...
        'Yes','No','Yes');
    switch selection
        case 'Yes'
            ubm = handles.database.ubm;
            gmm_models = handles.database.gmm_models;
            norders = handles.database.norders;
            fl = handles.database.fl;
            fh = handles.database.fh;
            Fs = handles.setupdata.Fs;
            curSignal = handles.setupdata.curSignal;
            switch norders
                case 3
                    curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'MDd0');
                case 2
                    curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'MD0');
                case 1
                    curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'M0');
            end
            curNormFeature = feanormalize(curmfccFeature,'w',Fs); % 'w' for warping or 'c' for cmvn
            map_tau = 10.0;
            config = 'mwv';
            curspeakerfile{1} = curNormFeature';
            handles.database.no_of_fe = handles.database.no_of_fe +1;
            gmm_models{handles.database.no_of_fe} = mapAdapt(curspeakerfile, ubm, map_tau, config);
            gmm_models{handles.database.no_of_fe}.name = per_name;
            save(handles.database.dataBasefileName,'ubm','gmm_models');
            
        case 'No'
            return
    end% end of switch
    
end % end for if to check duplicate name

handles.database.gmm_models = gmm_models;
set(handles.text_DataBaseNumber,'String',handles.database.no_of_fe);
handles.speakerName{handles.database.no_of_fe} = per_name;
set(handles.pm_SpeakerName,'String',...
    handles.speakerName)
guidata(hObject,handles)


% --- Executes on button press in bt_Train.
function bt_Train_Callback(hObject, eventdata, handles)
% hObject    handle to bt_Train (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Step0: Opening MATLAB pool
nworkers = 12;
nworkers = min(nworkers, feature('NumCores'));
isopen = matlabpool1('size')>0;
if ~isopen, matlabpool1(nworkers); end

selpath = uigetdir;
if selpath ~=0
    oldFolder = cd(selpath);
    waveFiles = dir('*.wav');
else
    return
end
hwt = waitbar(0,'please wait....');
nWav = length(waveFiles);
def = {'3';'0';'8'};
prompt = {'Enter order of melcepst:';...
    'Enter low frequency f_l(kHz):';...
    'Enter high frequency f_h(kHz):'};
dlg_title = 'Input for feature extraction ';
num_lines = 1;
answer = inputdlg(prompt,dlg_title,num_lines,def);

if ~isempty(answer)
    norders = str2num(answer{1});
    fl = str2num(answer{2});
    fh = str2num(answer{3});
    fl = fl*1e3;
    fh = fh*1e3;
    if fh<=fl
        errordlg('f_h should be larger than f_l!')
        return
    end
else
    return
end
trainSpeakerData = cell(1,nWav);
for ifile = 1:nWav
    curFileName = waveFiles(ifile).name;
    [signal,Fs] = audioread(curFileName);
    signal = signal - mean(signal);
    curSignal = signal/max(abs(signal));
    if norders == 1
        [mfccFeature] = melcepst(curSignal,[fl,fh,Fs],'M0');
    elseif norders == 2
        [mfccFeature] = melcepst(curSignal,[fl,fh,Fs],'MD0');
    elseif norders == 3
        [mfccFeature] = melcepst(curSignal,[fl,fh,Fs],'MDd0');
    end
    %     No_of_Gaussians = 12;
    normfeature = feanormalize(mfccFeature,'w',Fs); % 'w' for warping or 'c' for cmvn
    trainSpeakerData{ifile} = normfeature';
    %     [mu_train,sigma_train,c_train] = gmm_estimate(feature',No_of_Gaussians,20);
    %     per_name = curFileName(1:(end-4));
    %     if any(strcmp(per_name,handles.database.name)>0)  % find if the new name already exists in data base
    %         errordlg('name Already exists');
    %     else
    %         no_of_fe = handles.database.no_of_fe + 1;
    %         name = handles.database.name;
    %         fea = handles.database.fea;
    %         fea{no_of_fe,1} = mu_train;
    %         fea{no_of_fe,2} = sigma_train;
    %         fea{no_of_fe,3} = c_train;
    %         name(no_of_fe,1:length(per_name)) = per_name;
    %         cd(oldFolder)
    %         save(handles.database.dataBasefileName,'no_of_fe','name','fea');
    %         cd(selpath)
    %         handles.database.no_of_fe = no_of_fe;
    %         handles.database.name = name;
    %         handles.database.fea = fea;
    %         set(handles.text_DataBaseNumber,'String',handles.database.no_of_fe);
    %
    %     end % end for if to check duplicate name
    
    waitbar(ifile/nWav);
end
close(hwt);
%Step1: Create the universal background model from all the training speaker data
nmix = 32;           % In this case, we know the # of mixtures needed
final_niter = 10;
ds_factor = 1;              % down sampling factor
ubm = gmm_em(trainSpeakerData(:), nmix, final_niter, ds_factor, nworkers);
cd(oldFolder)
uisave({'ubm','norders','fl','fh'},'UBM-');
cd(selpath)
handles.database.ubm = ubm;
handles.database.norders = norders;
handles.database.fl = fl;
handles.database.fh = fh;

cd(oldFolder);
guidata(hObject,handles)

function varargout = matlabpool1(varargin)
varargout={};
arg1=varargin{1};
switch arg1
    case 'open'
        parpool(varargin{2:end});
    case 'close'
        delete(gcp('nocreate'));
    case  'size'
        p = gcp('nocreate'); % If no pool, do not create new one.
        if isempty(p)
            poolsize = 0;
        else
            poolsize = p.NumWorkers;
        end
        varargout={poolsize};
    otherwise %number of workers
        if ~isnumeric(arg1)
            arg1=num2str(arg1);
        end
        parpool(arg1);
end

function gmm = gmm_em(dataList, nmix, final_niter, ds_factor, nworkers, gmmFilename)
% fits a nmix-component Gaussian mixture model (GMM) to data in dataList
% using niter EM iterations per binary split. The process can be
% parallelized in nworkers batches using parfor.
%
% Inputs:
%   - dataList    : ASCII file containing feature file names (1 file per line)
%					or a cell array containing features (nDim x nFrames).
%					Feature files must be in uncompressed HTK format.
%   - nmix        : number of Gaussian components (must be a power of 2)
%   - final_iter  : number of EM iterations in the final split
%   - ds_factor   : feature sub-sampling factor (every ds_factor frame)
%   - nworkers    : number of parallel workers
%   - gmmFilename : output GMM file name (optional)
%
% Outputs:
%   - gmm		  : a structure containing the GMM hyperparameters
%					(gmm.mu: means, gmm.sigma: covariances, gmm.w: weights)
%
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

if ischar(nmix), nmix = str2double(nmix); end
if ischar(final_niter), final_niter = str2double(final_niter); end
if ischar(ds_factor), ds_factor = str2double(ds_factor); end
if ischar(nworkers), nworkers = str2double(nworkers); end

[ispow2, ~] = log2(nmix);
if ( ispow2 ~= 0.5 ),
    error('oh dear! nmix should be a power of two!');
end

if ischar(dataList) || iscellstr(dataList),
    dataList = load_data(dataList);
end
if ~iscell(dataList),
    error('Oops! dataList should be a cell array!');
end

nfiles = length(dataList);

fprintf('\n\nInitializing the GMM hyperparameters ...\n');
[gm, gv] = comp_gm_gv(dataList);
gmm = gmm_init(gm, gv);

% gradually increase the number of iterations per binary split
% mix = [1 2 4 8 16 32 64 128 256 512 1024];
niter = [1 2 4 4  4  4  6  6   10  10  15];
niter(log2(nmix) + 1) = final_niter;

mix = 1;
while ( mix <= nmix )
    if ( mix >= nmix/2 ), ds_factor = 1; end % not for the last two splits!
    fprintf('\nRe-estimating the GMM hyperparameters for %d components ...\n', mix);
    for iter = 1 : niter(log2(mix) + 1)
        fprintf('EM iter#: %d \t', iter);
        N = 0; F = 0; S = 0; L = 0; nframes = 0;
        tim = tic;
        parfor (ix = 1 : nfiles, nworkers)
            %         for ix = 1 : nfiles,
            [n, f, s, l] = expectation(dataList{ix}(:, 1:ds_factor:end), gmm);
            N = N + n; F = F + f; S = S + s; L = L + sum(l);
            nframes = nframes + length(l);
        end
        tim = toc(tim);
        fprintf('[llk = %.2f] \t [elaps = %.2f s]\n', L/nframes, tim);
        gmm = maximization(N, F, S);
    end
    if ( mix < nmix ),
        gmm = gmm_mixup(gmm);
    end
    mix = mix * 2;
end

if ( nargin == 6 ),
    fprintf('\nSaving GMM to file %s\n', gmmFilename);
    % create the path if it does not exist and save the file
    path = fileparts(gmmFilename);
    if ( exist(path, 'dir')~=7 && ~isempty(path) ), mkdir(path); end
    save(gmmFilename, 'gmm');
end

function data = load_data(datalist)
% load all data into memory
if ~iscellstr(datalist)
    fid = fopen(datalist, 'rt');
    filenames = textscan(fid, '%s');
    fclose(fid);
    filenames = filenames{1};
else
    filenames = datalist;
end
nfiles = size(filenames, 1);
data = cell(nfiles, 1);
for ix = 1 : nfiles,
    data{ix} = htkread(filenames{ix});
end

function [gm, gv] = comp_gm_gv(data)
% computes the global mean and variance of data
nframes = cellfun(@(x) size(x, 2), data, 'UniformOutput', false);
nframes = sum(cell2mat(nframes));
gm = cellfun(@(x) sum(x, 2), data, 'UniformOutput', false);
gm = sum(cell2mat(gm'), 2)/nframes;
gv = cellfun(@(x) sum(bsxfun(@minus, x, gm).^2, 2), data, 'UniformOutput', false);
gv = sum(cell2mat(gv'), 2)/( nframes - 1 );

function gmm = gmm_init(glob_mu, glob_sigma)
% initialize the GMM hyperparameters (Mu, Sigma, and W)
gmm.mu    = glob_mu;
gmm.sigma = glob_sigma;
gmm.w     = 1;

function [N, F, S, llk] = expectation(data, gmm)
% compute the sufficient statistics
[post, llk] = postprob(data, gmm.mu, gmm.sigma, gmm.w(:));
N = sum(post, 2)';
F = data * post';
S = (data .* data) * post';

function [post, llk] = postprob(data, mu, sigma, w)
% compute the posterior probability of mixtures for each frame
post = lgmmprob(data, mu, sigma, w);
llk  = logsumexp(post, 1);
post = exp(bsxfun(@minus, post, llk));

function logprob = lgmmprob(data, mu, sigma, w)
% compute the log probability of observations given the GMM
ndim = size(data, 1);
C = sum(mu.*mu./sigma) + sum(log(sigma));
D = (1./sigma)' * (data .* data) - 2 * (mu./sigma)' * data  + ndim * log(2 * pi);
logprob = -0.5 * (bsxfun(@plus, C',  D));
logprob = bsxfun(@plus, logprob, log(w));

function y = logsumexp(x, dim)
% compute log(sum(exp(x),dim)) while avoiding numerical underflow
xmax = max(x, [], dim);
y    = xmax + log(sum(exp(bsxfun(@minus, x, xmax)), dim));
ind  = find(~isfinite(xmax));
if ~isempty(ind)
    y(ind) = xmax(ind);
end

function gmm = maximization(N, F, S)
% ML re-estimation of GMM hyperparameters which are updated from accumulators
w  = N / sum(N);
mu = bsxfun(@rdivide, F, N);
sigma = bsxfun(@rdivide, S, N) - (mu .* mu);
sigma = apply_var_floors(w, sigma, 0.1);
gmm.w = w;
gmm.mu= mu;
gmm.sigma = sigma;

function sigma = apply_var_floors(w, sigma, floor_const)
% set a floor on covariances based on a weighted average of component
% variances
vFloor = sigma * w' * floor_const;
sigma  = bsxfun(@max, sigma, vFloor);
% sigma = bsxfun(@plus, sigma, 1e-6 * ones(size(sigma, 1), 1));

function gmm = gmm_mixup(gmm)
% perform a binary split of the GMM hyperparameters
mu = gmm.mu; sigma = gmm.sigma; w = gmm.w;
[ndim, nmix] = size(sigma);
[sig_max, arg_max] = max(sigma);
eps = sparse(0 * mu);
eps(sub2ind([ndim, nmix], arg_max, 1 : nmix)) = sqrt(sig_max);
% only perturb means associated with the max std along each dim
mu = [mu - eps, mu + eps];
% mu = [mu - 0.2 * eps, mu + 0.2 * eps]; % HTK style
sigma = [sigma, sigma];
w = [w, w] * 0.5;
gmm.w  = w;
gmm.mu = mu;
gmm.sigma = sigma;


% --- Executes on button press in bt_Modeling.
function bt_Modeling_Callback(hObject, eventdata, handles)
% hObject    handle to bt_Modeling (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

ubm = handles.database.ubm;
norders = handles.database.norders;
fl = handles.database.fl;
fh = handles.database.fh;
% [ndims,~] = size(ubm.mu);
selpath = uigetdir;
if selpath ~=0
    oldFolder = cd(selpath);
    waveFiles = dir('*.wav');
else
    return
end
map_tau = 10.0;
config = 'mwv';
nSpeaker = length(waveFiles);
curspeakerfile = cell(1,1);
gmm_models = cell(1,nSpeaker);
cd(selpath)
hwt = waitbar(0,'Speaker model....');
for ispeaker = 1:nSpeaker
    curFileName = waveFiles(ispeaker).name;
    [signal,Fs] = audioread(curFileName);
    signal = signal - mean(signal);
    curSignal = signal/max(abs(signal));
    if norders == 3
        curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'MDd0');
    elseif norders == 2
        curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'MD0');
    elseif norders == 1
        curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'M0');
    else
    end
    curNormFeature = feanormalize(curmfccFeature,'w',Fs); % 'w' for warping or 'c' for cmvn
    curspeakerfile{1} = curNormFeature';
    gmm_models{ispeaker} = mapAdapt(curspeakerfile, ubm, map_tau, config);
    gmm_models{ispeaker}.name = curFileName(1:end-4);
    %     % UBM likelihood
    %     likelihoodindex = zeros(1,UBM_no_of_fe);
    %     for i = 1:UBM_no_of_fe
    %         mu_t = UBM_fea{i,1};
    %         sigma_t = UBM_fea{i,2};
    %         c_t = UBM_fea{i,3};
    %         [~,likelihoodY] = lmultigauss(curNormFeature',mu_t,sigma_t,c_t);
    %         likelihoodindex(i) = mean(likelihoodY);
    %     end
    % %     likelihoodUBM(ispeaker,:) = likelihoodindex;
    %     [~,iadapt(ispeaker)] = max(likelihoodindex);
    waitbar(ispeaker/nSpeaker,hwt,'Speaker modelling....');
end
handles.database.gmm_models = gmm_models;
close(hwt);
cd(oldFolder);
uisave({'gmm_models','ubm','norders','fl','fh'},'Spkmodel-')
guidata(hObject,handles)

function gmm = mapAdapt(dataList, ubmFilename, tau, config, gmmFilename)
% MAP-adapts a speaker specific GMM gmmFilename from UBM ubmFilename using
% features in dataList. The MAP relevance factor can be specified via tau.
% Adaptation of all GMM hyperparameters are supported.
%
% Inputs:
%   - dataList    : ASCII file containing adaptation feature file name(s)
%                   or a cell array containing feature(s). Feature files
%					must be in uncompressed HTK format.
%   - ubmFilename : file name of the UBM or a structure containing
%					the UBM hyperparameters that is,
%					(ubm.mu: means, ubm.sigma: covariances, ubm.w: weights)
%   - tau         : the MAP adaptation relevance factor (19.0)
%   - config      : any sensible combination of 'm', 'v', 'w' to adapt
%                   mixture means (default), covariances, and weights
%   - gmmFilename : the output speaker specific GMM file name (optional)
%
% Outputs:
%   - gmm		  : a structure containing the GMM hyperparameters
%					(gmm.mu: means, gmm.sigma: covariances, gmm.w: weights)
%
% References:
%   [1] D.A. Reynolds, T.F. Quatieri, and R.B. Dunn, "Speaker verification
%       using adapted Gaussian mixture models," Digital Signal Process.,
%       vol. 10, pp. 19-41, Jan. 2000.
%
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

if ( nargin < 3 ),
    tau = 19.0; % MAP adaptation relevance factor
end
if ( nargin < 4 ), config = ''; end;

if ischar(tau), tau = str2double(tau); end

if isempty(config), config = 'm'; end

if ischar(ubmFilename),
    tmp = load(ubmFilename);
    ubm = tmp.gmm;
elseif isstruct(ubmFilename),
    ubm = ubmFilename;
else
    error('oh dear! ubmFilename should be either a string or a structure!');
end

gmm = ubm;

if ischar(dataList) || iscellstr(dataList),
    dataList = load_data(dataList);
end
if ~iscell(dataList),
    error('Oops! dataList should be a cell array!');
end
nfiles = length(dataList);

N = 0; F = 0; S = 0;
parfor file = 1 : nfiles,
    [n, f, s] = expectation(dataList{file}, ubm);
    N = N + n; F = F + f; S = S + s;
end

if any(config == 'm'),
    alpha = N ./ (N + tau); % tarde-off between ML mean and UBM mean
    m_ML = bsxfun(@rdivide, F, N);
    m = bsxfun(@times, ubm.mu, (1 - alpha)) + bsxfun(@times, m_ML, alpha);
    gmm.mu = m;
end

if any(config == 'v'),
    alpha = N ./ (N + tau);
    v_ML = bsxfun(@rdivide, S, N);
    v = bsxfun(@times, (ubm.sigma+ubm.mu.^2), (1 - alpha)) + bsxfun(@times, v_ML, alpha) - (m .* m);
    gmm.sigma = v;
end

if any(config == 'w'),
    alpha = N ./ (N + tau);
    w_ML = N / sum(N);
    w = bsxfun(@times, ubm.w, (1 - alpha)) + bsxfun(@times, w_ML, alpha);
    w = w / sum(w);
    gmm.w = w;
end

if ( nargin == 5 ),
    % create the path if it does not exist and save the file
    path = fileparts(gmmFilename);
    if ( exist(path, 'dir')~=7 && ~isempty(path) ), mkdir(path); end
    save(gmmFilename, 'gmm');
end


function [data, frate, feakind] = htkread(filename)
% read features with HTK format (uncompressed)
fid = fopen(filename, 'rb', 'ieee-be');
nframes = fread(fid, 1, 'int32'); % number of frames
frate   = fread(fid, 1, 'int32'); % frame rate in nano-seconds unit
nbytes  = fread(fid, 1, 'short'); % number of bytes per feature value
feakind = fread(fid, 1, 'short'); % 9 is USER
ndim = nbytes / 4; % feature dimension (4 bytes per value)
data = fread(fid, [ndim, nframes], 'float');
fclose(fid);


% --- Executes on button press in bt_EER.
function bt_EER_Callback(hObject, eventdata, handles)
% hObject    handle to bt_EER (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if isempty(handles.database.ubm) || isempty(handles.database.gmm_models)
    errordlg('Oh dear, without UBM and Speakermodels, I can not do this!')
    return
end
ubm = handles.database.ubm;
gmm_models = handles.database.gmm_models;
norders = handles.database.norders;
fl = handles.database.fl;
fh = handles.database.fh;
nSpeaker = length(gmm_models);
% [ndims,~] = size(ubm.mu);
testpath = uigetdir;
if testpath ~= 0
    oldpath = cd(testpath);
    testWaveFiles = dir('*.wav');
else
    return
end
nTests = length(testWaveFiles);

% Performace measure- Detection error trade-off(DET) curve
def = {'-1:0.001:1'};
prompt = {'Enter threshold measure:'};
dlg_title = 'Input for performace measure ';
num_lines = 1;
answer = inputdlg(prompt,dlg_title,num_lines,def);

if ~isempty(answer)
    thresholdmeasure = str2num(answer{1});
else
    return
end

hwt = waitbar(0,'please wait....');
tests = cell(1,nTests);
testsname = cell(1,nTests);
cd(testpath)
for iTest = 1:nTests
    curFileName = testWaveFiles(iTest).name;
    [signal,Fs] = audioread(curFileName);
    signal = signal - mean(signal);
    curSignal = signal/max(abs(signal));
    switch norders
        case 1
            curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'M0');
        case 2
            curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'MD0');
        case 3
            curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'MDd0');
    end
    curNormFeature = feanormalize(curmfccFeature,'w',Fs); % 'w' for warping or 'c' for cmvn
    tests{iTest} = curNormFeature';
    testsname{iTest} = curFileName(1:end-4);
    waitbar(0.3*iTest/nTests,hwt,'Performance measuring....');
end

llr = zeros(nSpeaker, nTests);
trueresults = zeros(nSpeaker, nTests);
nMiss = zeros(1,length(thresholdmeasure));
nFalseAccept = zeros(1,length(thresholdmeasure));
for ispeaker = 1 : nSpeaker,
    gmm = gmm_models{1,ispeaker};
    labelspeaker = gmm_models{1, ispeaker}.name(1:end-4);
    for iTest = 1:nTests
        fea = tests{1,iTest};
        ubm_llk = compute_llk(fea, ubm.mu, ubm.sigma, ubm.w(:));
        gmm_llk = compute_llk(fea, gmm.mu, gmm.sigma, gmm.w(:));
        llr(ispeaker,iTest) = mean(gmm_llk - ubm_llk);
        p = strfind(testsname{iTest},'-');
        if ~isempty(p)
            labeltest = testsname{iTest}(1:(p(end)-1));
        end
        trueresults(ispeaker,iTest) = contains(labeltest,labelspeaker);
    end
    waitbar(0.3+0.5*ispeaker/nSpeaker,hwt,'Performance measuring....');
end
for ithreshold = 1:length(thresholdmeasure)
    threshold = thresholdmeasure(ithreshold);
    testresult = llr > threshold;
    nMiss(ithreshold) = sum(sum(trueresults)) - sum(sum(trueresults.*testresult));
    nFalseAccept(ithreshold) = sum(sum(testresult.*(~trueresults)));
    waitbar(0.8+0.2*ithreshold/length(thresholdmeasure),hwt,'Performance measuring....');
end

nTotalTest = nSpeaker*nTests;
FAR = nFalseAccept./nTotalTest;  % False Acceptance Rate
MR = nMiss./nTotalTest;  % Miss Rate
[~,indEER] = min(abs(FAR-MR));
bestTestResult = llr > thresholdmeasure(indEER);
fasleResult = trueresults - bestTestResult;

disp(['total test #:  ' num2str(nTotalTest)])
disp(['Miss #:  ' num2str(nMiss(indEER))])
for ispeaker = 1 : nSpeaker,
    labelspeaker = gmm_models{1, ispeaker}.name(1:end-4);
    for iTest = 1:nTests
        labeltest = testsname{iTest};
        if fasleResult(ispeaker,iTest)==1
        disp(['Miss ' labeltest ' To ' labelspeaker;])
        end
    end
end
disp(['FalseAccept #:  ' num2str(nFalseAccept(indEER))])
for ispeaker = 1 : nSpeaker,
    labelspeaker = gmm_models{1, ispeaker}.name(1:end-4);
    for iTest = 1:nTests
        labeltest = testsname{iTest};
        if fasleResult(ispeaker,iTest)==-1
        disp(['False accept ' labeltest ' To ' labelspeaker;])
        end
    end
end
EER = FAR(indEER);

close(hwt)
axes(handles.axes_likelihood)
plot(FAR*1e2,MR*1e2,'*','Linewidth',1.5)
title(['EER = ' num2str(round(EER*1e4)/1e2) '% at threshold = ' num2str(thresholdmeasure(indEER))])
xlabel('False Acceptance Rate(in%)')
ylabel('Miss Rate(in%)')
cd(oldpath)
figure;
plot(FAR*1e2,MR*1e2,'*','Linewidth',1.5)
title(['EER = ' num2str(round(EER*1e4)/1e2) '% at threshold = ' num2str(thresholdmeasure(indEER))])
xlabel('False Acceptance Rate(in%)')
ylabel('Miss Rate(in%)')

function llk = compute_llk(data, mu, sigma, w)
% compute the posterior probability of mixtures for each frame
post = lgmmprob(data, mu, sigma, w);
llk  = logsumexp(post, 1);


% --- Executes on button press in bt_LoadUBM.
function bt_LoadUBM_Callback(hObject, eventdata, handles)
% hObject    handle to bt_LoadUBM (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename, pathname] = uigetfile({'UBM-*.mat';...
    '*.*'},'select a UBM file to load');
if filename == 0
    errordlg('ERROR! No file selected!');
    return;
end
handles.database = load([pathname filename]);
%  = ubm;
guidata(hObject,handles);


% --- Executes during object creation, after setting all properties.
function axes_Spectrogram_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes_Spectrogram (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes_Spectrogram


% --- Executes on button press in bt_SpeakerVerification.
function bt_SpeakerVerification_Callback(hObject, eventdata, handles)
% hObject    handle to bt_SpeakerVerification (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

THR = 0.016;
if isempty(handles.database.ubm) || isempty(handles.database.gmm_models)
    errordlg('Oh dear, without UBM and Speakermodels, I can not do this!')
    return
end
ubm = handles.database.ubm;
gmm_models = handles.database.gmm_models;
norders = handles.database.norders;
fl = handles.database.fl;
fh = handles.database.fh;
iSignSpeaker = get(handles.pm_SpeakerName,'Value');
Fs = handles.setupdata.Fs;
curSignal = handles.setupdata.curSignal;

switch norders
    case 3
        curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'MDd0');
    case 2
        curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'MD0');
    case 1
        curmfccFeature = melcepst(curSignal,[fl,fh,Fs],'M0');
end
curNormFeature = feanormalize(curmfccFeature,'w',Fs); % 'w' for warping or 'c' for cmvn
gmm = gmm_models{1,iSignSpeaker};
ubm_llk = compute_llk(curNormFeature', ubm.mu, ubm.sigma, ubm.w(:));
gmm_llk = compute_llk(curNormFeature', gmm.mu, gmm.sigma, gmm.w(:));
likelihoodindex = mean(gmm_llk - ubm_llk);
if likelihoodindex > THR
    set(handles.text_quality,'string','Accept','visible','on');
    set(handles.text_likelihood,'string',likelihoodindex,'visible','on');
else
    set(handles.text_quality,'string','Reject','visible','on');
    set(handles.text_likelihood,'string',likelihoodindex,'visible','on');
end


% --- Executes on selection change in pm_SpeakerName.
function pm_SpeakerName_Callback(hObject, eventdata, handles)
% hObject    handle to pm_SpeakerName (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pm_SpeakerName contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pm_SpeakerName


% --- Executes during object creation, after setting all properties.
function pm_SpeakerName_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pm_SpeakerName (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in bt_STFT.
function bt_STFT_Callback(hObject, eventdata, handles)
% hObject    handle to bt_STFT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Fs = handles.setupdata.Fs;
curSignal = handles.setupdata.curSignal;
curSignal1 = handles.setupdata.curSignal1;
axes(handles.axes_Spectrogram)
cla(gca)
spectrumplot(curSignal,Fs)
%second STFT
axes(handles.axes24)
cla(gca)
spectrumplot(curSignal1,Fs)
if Fs == 80e3
    ylim([0,40])
else
    ylim([0,Fs/2e3])
end
