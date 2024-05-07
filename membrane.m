N = 40; 
M = 100; dte = 2*pi/M; t = dte*(0:M)';
h=1/N;
ta=h^2;
r=h*[0:N];
[rr,tt] = meshgrid(r,t);
[xx,yy] = pol2cart(tt,rr);

%u0=sech(rr).^2;
u0 = 1./(rr.^2+1).*cos(5*tt);
u0=-rr.^2.*sin(tt/2) + sin(6*tt).*cos(tt/2).^2;

u1=u0;
hta2=ta^2/h^2;
nn=[0:M/2 -M/2+1:-1].^2;nn=nn';
for time=1:5000
    for col=1:N+1
        u1t(2:M+1,col)=ifft(nn.*fft(u1(2:M+1,col)));
        u1t(1,col)=u1t(M+1,col);
    end
    u2(:,2:N)=-u0(:,2:N)+2*u1(:,2:N)...
        +hta2*(u1(:,1:N-1)-2*u1(:,2:N)+u1(:,3:N+1))...
        +hta2/2*(u1(:,3:N+1)-u1(:,1:N-1))*diag(1./[1:N-1],0)...
        -hta2*u1t(:,2:N)*diag(1./[1:N-1].^2,0);
    u2(:,N+1)=zeros(M+1,1);
    u2(:,1)=ones(M+1,1)*mean(u2(:,2));
    u0=u1;
    u1=u2;
    if rem(time,10)==0
        surf(xx,yy,u2),colormap jet, shading interp
        view(30,60)
        drawnow
    end
end