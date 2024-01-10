function [w, beta, residp, residn, totalviolation, X, y, Z]=wdwdls(Ap, An, paramstruct);
%
%WDWDLS Lingsong's Weighted Distance Weighted Discrimination
%     written by Lingsong Zhang, which is very similar to Marron's original code
%     this routine is just for practice.
%
% DWDLS uses a newer version of SDPT3, and thus please temporarily rename the
% sqlparameters in the directory of Marron's DWD code if you have it. Or please
% remove the directory from the matlab path.
% 
%
%  The weighted DWD method was proposed in Qiao, Zhang, Liu, Todd and
%  Marron (2008). 
% 
% Usage:
%
%    [w, beta, residp, residn, totalviolation, X, y, Z]=sdwd2ls(Ap, An, paramstruct);
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
%
%    DWDpar      this corresponds to the penalty coefficient of the
%                misclassification
%
%    obsweight   the weights assigned to each observations, in order to 
%                solve the unbalance issue for some data sets. Default is
%                all ones, i.e., each observation contributes the same in
%                finding the classifier.
%
%                Note: please put the case in the first part, and the
%                control in the second part. Please pay attention to the
%                order in your data set. The length of this input vector
%                should be n1+n2, where n1 is the sample size of the cases,
%                and n2 is the sampel size of the controls.
%
% Outputs:
%
%    w           the normal vector
%
%    b           the location vector
%
%(c)Copyright Lingsong Zhang (lingsong@purdue.edu)
% 
% 2007-11-13 original code for DWD
% 2009-07-15 adapted for wDWD method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<2;
    error('Please input both the case and the control!');
end;

DWDpar=100;

[dp, np]=size(Ap);
[dn, nn]=size(An);
n=np+nn;

obsweight=ones(n, 1);

if dp~=dn;
   disp('The dimensions of the training sets do not match');   
end;

d=dp;

if nargin>2;%we need to update some inputs
    if isfield(paramstruct, 'DWDpar');
        DWDpar=getfield(paramstruct, 'DWDpar');
    end;
    
    if isfield(paramstruct, 'obsweight');
        obsweight=getfield(paramstruct, 'obsweight');
    end;
end;

if length(obsweight)==2;
    disp('The observations in each class are assigned with the same weights');
    obsweight=[obsweight(1)*ones(np, 1); obsweight(2)*ones(nn, 1)];
end;


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

%ctemp=[1; -1; 0];

if d>n
     cq=zeros(n+1, 1);
else
    cq=zeros(d+1, 1);
end;

for (i=1:n);
  cq=[cq; obsweight(i); -obsweight(i); 0];
end;

%cl=penalty*ones(n, 1);
cl=penalty*obsweight; %this is adjusted for weighted DWD
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
