tobs=1e4;
Mc=3e-4;
f0=120;
fmax=200;

G = 6.67e-11;
c=3e8;
lambda = 96/5*pi^(8/3)*(G/c^3)^(5/3);

dt = 1./(40*fmax);
nt = round(tobs/dt);
t = (0 : nt-1)*dt;

Mc = Mc * 2e30;
chi0 = lambda * Mc^(5/3)*f0^(8/3);

% generate signal and add to noise

%phi0=2*pi*f0*t;
phi = -2*pi*f0*3/5*(1-8./3.*chi0.*t).^(5/8)/chi0;
phi = mod(phi,2*pi);
signal = 1*exp(1j*phi);

nh = 50;
noise = nh * randn(1,nt);

data = signal + noise;
%data = signal;

sdata = fft(data);
P2 = abs(sdata/nt);
P1 = P2(1:nt/2+1);
P1(2:end-1) = 2*P1(2:end-1);
freq = 1./dt*(0:(nt/2))/nt;
figure;plot(freq,P1); 

toff = 0.0;
chi = chi0;
dtout = dt * 10;

tt = -3/5*(1-8./3.*chi.*t).^(5/8)/chi;
tt = tt/dtout;
tt1 = floor(tt);
ii = find(diff(tt1));
out = data(ii+1);

out = out(1:numel(out)-1);
nt = numel(out);
sdatacorr = fft(out);
P2 = abs(sdatacorr/nt);
P1 = P2(1:nt/2+1);
P1(2:end-1) = 2*P1(2:end-1);
freq = 1./(dtout)*(0:(nt/2))/nt;
figure;plot(freq,P1);title('data corr'); 