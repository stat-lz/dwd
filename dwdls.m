function [w, beta, residp, residn, totalviolation, X, y, Z]=dwdls(Ap, An, DWDpar);
%DWDLS Distance Weighted Discrimination
%     written by Lingsong Zhang, which is very similar to Marron's original code
%     this is also just for practice.
%
% Usage:
%
%    [w, beta, residp, residn, totalviolation, X, y, Z]=sdwd2ls(Ap, An, DWDpar);
%
%
% Inputs:
%
%    Ap          input matrix for case (patient group)
%
%    An          input matrix for control (normal group)
%
%                Note that for the above two matrices, each column
%                corresponds to each subject, and the rows are covariates
%
%    DWDpar      this corresponds to the penalty coefficient of the
%                misclassification
%
%
% Outputs:
%
%    w           the normal vector
%
%
%
%(c)Copyright Lingsong Zhang (zhang@hsph.harvard.edu)
% 
% 2007-11-13 modification
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin==2;
   DWDpar=100;
end;

[dp, np]=size(Ap);
[dn, nn]=size(An);
n=np+nn;


if dp~=dn;
   disp('The dimensions of the training sets do not match');   
end;

d=dp;

if nargin==2;
    DWDpar=100;
elseif nargin==3;
    t=sqrt(d);
end;

dsave=d;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The following part is copied from Marron's DWD1SM.m file to get the tuning parameter
%
%  Compute median of pairwise distances squared between classes
%
vpwdist2 = [] ;
for ip = 1:np ;
  pwdist2 = sum((vec2matSM(Ap(:,ip),nn) - An).^2,1) ;
  vpwdist2 = [vpwdist2 pwdist2] ;
end ;
medianpwdist2 = median(vpwdist2) ;

penalty = DWDpar / medianpwdist2;
%penalty=49;
    %  threshfact "makes this large", 
    %  and 1 / medianpwdist2 "puts on correct scale"

% Problems, please refer to Marron et al, (2004) DWD paper.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SX=[Ap, -An];

SXnew = SX';



%define the number of variable, and the number of constraints
if d>n;
    nv=2+5*n;
    nc=2*n+1;
    [Q,R] = qr(SX,0);
    SXnew=R';
else
    nv=2+d+4*n; %omega, w, beta, rho, sigma, tau, xi, Omega, s, p, q  
    nc=2*n+1;
end;

y=[ones(np, 1); -1*ones(nn, 1)]; %n



%three blocks needed

blk=cell(3, 2);
blk{1, 1}='q';
if d>n;
    blk{1, 2}=[n+1, 3*ones(1, n)];
else
    blk{1, 2}=[d+1, 3*ones(1, n)];
end;

blk{2, 1}='l';
blk{2, 2}=n;

blk{3, 1}='u';
blk{3, 2}=1;

Avec=cell(3, 1);

% if d>n;
%     Aq=sparse(nc, 1+n+3*n);
%     
%     Aq(n+1, 1)=1;
%     Aq(1:n, 2:(n+1))=SXnew;
%     Aq((2*(n+1)+1):nc, 2:(n+1))=[Q; -Q];
%     
%     for (k=1:n);
%         Aq(k, ((k-1)*3+n+2))=-1;
%         Aq(k, ((k-1)*3+n+3))=-1;
%         Aq((n+k+1), ((k-1)*3+n+4))=1;
%     end;
% else;

if d>n;
  Aq=sparse(nc, 1+n+3*n);
  
  Aq(n+1, 1)=1;
  Aq(1:n, 2:(n+1))=SXnew;
%  Aq((2*(n+1)):(nc-1), 2:(n+1))=Q;
  
  for (k=1:n);
      Aq(k, ((k-1)*3+n+2))=-1;
      Aq(k, ((k-1)*3+n+3))=-1;
      Aq((n+k+1), ((k-1)*3+n+4))=1;
  end;
  
else
  Aq=sparse(nc, 1+d+3*n);
    
    Aq(n+1, 1)=1;
    Aq(1:n, 2:(d+1))=SXnew;
%    Aq((2*(n+1)):(nc-1), 2:(d+1))=speye(d, d);
    
    for (k=1:n);
        Aq(k, ((k-1)*3+d+2))=-1;
        Aq(k, ((k-1)*3+d+3))=-1;
        Aq((n+k+1), ((k-1)*3+d+4))=1;
    end;
end;

Al=sparse(nc, n);
Al(:, 1:n)=speye(nc, n);

Au=[y; zeros(n+1, 1)];

Avec{1, 1}=Aq;
Avec{2, 1}=Al;
Avec{3, 1}=Au;

ctemp=[1; -1; 0];

if d>n
     cq=zeros(n+1, 1);
else
    cq=zeros(d+1, 1);
end;

for (i=1:n);
  cq=[cq; ctemp];
end;

cl=penalty*ones(n, 1);;
cu=0;

C=cell(3, 1);
C{1, 1}=cq;
C{2, 1}=cl;
C{3, 1}=cu;

b=[zeros(n, 1); ones(1+n, 1)];


%the following use sqlp to solve the problem
OPTIONS=sqlparameters;
OPTIONS.printlevel=0;
OPTIONS.maxit=200;
[X0,lambda0,Z0] = infeaspt(blk,Avec,C,b, 2, 1e5);
[obj, X, y, Z, info]=sqlp(blk, Avec, C, b, OPTIONS, X0, lambda0, Z0);
X1=X{1}; %this is the argument of the optimization for q part
X2=X{2}; %for l part
X3=X{3}; %for u part

omega=X1(1);
X1length=length(X1);
 if d>n
     wtemp=X1(2:(n+1));
     w=Q*wtemp;
 else
    w=X1(2:(d+1));
end;
beta=X3;
