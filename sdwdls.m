function [w, beta]=sdwdls(Ap, An, paramstruct);
%SDWDLS L1 Sparse and Adaptive Sparse Distance Weighted Discrimination
%
% Usage:
%
%    [w, beta]=sdwdls(Ap, An, paramstruct);
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
%    paramstruct
%
%         
%      a Matlab structure of input parameters
%                    Use: "help struct" and "help datatypes" to
%                         learn about these.
%                    Create one, using commands of the form:
%
%       paramstruct = struct('field1',values1,...
%                            'field2',values2,...
%                            'field3',values3) ;
%
%                          where any of the following can be used,
%                          these are optional, misspecified values
%                          revert to defaults
%    fields             Value
%
%    DWDpar        this corresponds to the penalty coefficient of the
%                  misclassification (default value is 100)
%
%    t             the contraint of L1 or adaptive L1 norm of the normal vector of the
%                  separating hyperplane. the range should be between 0 and
%                  p, where p is the number of variables in the data
%                  (default value is p for simplicity and yield the full model)
%
%    tol           the threshold to set all coefficient (abs) below as 0. default 
%                  value is 10^(-6).
%
%    istandardize  whether the final result is standardized or not, default
%                  is 0
%
%    iadaptive     0     the usual L1-SDWD (default value)
%                  1     the adaptive L1-SDWD
%
% 
% Outputs:
%
%    w           the normal vector
%
%    b            the location vector
%
%
%
% Dependent matlab functions:
%
%    dwdls.m    an experimental routine of dwd method
%    SDPT3 package, which can be downloaded from 
%          http://www.math.nus.edu.sg/~mattohkc/sdpt3.html
%
%(c)Copyright Lingsong Zhang (zhang@hsph.harvard.edu)
% 
% 2007-11-13 modification
% 2008-08-25 second modification
% 2008-11-01 third modification
% 2009-07-14 Combining SDWD and ASDWD together
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   istandardize=0;
   tol=1e-6;
   DWDpar=100;
   iadaptive=0;

   if nargin<2;
       error('Please input both the cases and the controls!');
   end;
   
[dp, np]=size(Ap);
[dn, nn]=size(An);
n=np+nn;

if dp~=dn;
   disp('The dimensions of the training sets do not match');   
end;

d=dp;

t=d;

if nargin>2;
    if isfield(paramstruct, 'DWDpar');
        DWDpar=getfield(paramstruct, 'DWDpar');
    end;

    if isfield(paramstruct, 't');
        t=getfield(paramstruct, 't');
    end;
    
    if isfield(paramstruct, 'tol');
        tol=getfield(paramstruct, 'tol');
    end;
    
    if isfield(paramstruct, 'istandardize');
        istandardize=getfield(paramstruct, 'istandardize');
    end;    

    if isfield(paramstruct, 'iadaptive');
	iadaptive=getfield(paramstruct, 'iadaptive');
    end;
end;


%if nargin==2;
%    DWDpar=100;
%    t=d;
%elseif nargin==3;
%    t=d;
%end; %do not understand why we have this

dsave=d;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The following part is copied from Marron's DWD1SM.m file to get the tuning parameter
%
%  Compute median of pairwise distances squared between classes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%
%%% First calculate the regular DWD based on the DWDpar
%%%

w_dwd=dwdls(Ap, An, DWDpar);

if iadaptive==1;
    adaptvec=(1./abs(w_dwd));
elseif iadaptive==0;
    adaptvec=ones(d, 1);
end;

if nargin<=2 | ~isfield(paramstruct, 't');
        if iadaptive==0;
            t=sum(abs(w_dwd));
        end;
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SX=[Ap, -An];

SXnew = SX';



%define the number of variable, and the number of constraints
if d>n;
    nv=3+2*d+5*n;
    nc=2*n+d+2;
    [Q,R] = qr(SX,0);
    SXnew=R';
else
    nv=3+3*d+4*n; %omega, w, beta, rho, sigma, tau, xi, Omega, s, p, q  
    nc=2*n+d+2;
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
blk{2, 2}=2*d+n+1;

blk{3, 1}='u';
blk{3, 2}=1;

Avec=cell(3, 1);

if d>n;
  Aq=sparse(nc, 1+n+3*n);
  
  Aq(n+1, 1)=1;
  Aq(1:n, 2:(n+1))=SXnew;
%  Aq((2*(n+1)):(nc-1), 2:(n+1))=adaptmat*Q;
  Aq((2*(n+1)):(nc-1), 2:(n+1))=Q;
   
  for (k=1:n);
      Aq(k, ((k-1)*3+n+2))=-1;
      Aq(k, ((k-1)*3+n+3))=-1;
      Aq((n+k+1), ((k-1)*3+n+4))=1;
  end;
  
else
  Aq=sparse(nc, 1+d+3*n);
    
    Aq(n+1, 1)=1;
    Aq(1:n, 2:(d+1))=SXnew;
    Aq((2*(n+1)):(nc-1), 2:(d+1))=speye(d, d);
    
    for (k=1:n);
        Aq(k, ((k-1)*3+d+2))=-1;
        Aq(k, ((k-1)*3+d+3))=-1;
        Aq((n+k+1), ((k-1)*3+d+4))=1;
    end;
end;

Al=sparse(nc, (2*d+n+1));
Al(:, 1:n)=speye(nc, n);
Al(2*n+2, (n+1):(n+d))=ones(1, d);
Al(((2*n+2):(nc-1)), (n+d+1):(n+2*d))=speye(d, d);
Al(((2*n+2):(nc-1)), (n+1:n+d))=-speye(d, d);
Al(nc, (n+1):(n+2*d))=[adaptvec' adaptvec'];
Al(nc, 2*d+n+1)=1;

Au=[y; zeros(n+d+2, 1)];

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

cl=[penalty*ones(n, 1); zeros(2*d+1, 1)];
cu=0;

C=cell(3, 1);
C{1, 1}=cq;
C{2, 1}=cl;
C{3, 1}=cu;

b=[zeros(n, 1); ones(1+n, 1); zeros(d, 1); t];


%the following use sqlp to solve the problem
OPTIONS=sqlparameters;
OPTIONS.printlevel=0;
OPTIONS.maxit=2000;
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

w(abs(w)<=tol)=0;

if istandardize==1;
beta=beta/norm(w);
w=w./norm(w);
end;

residp=obj(1);
