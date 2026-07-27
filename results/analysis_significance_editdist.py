# Bang chung thong ke cho goi sua Chuong 4 (E1/E2/E3) — chay tu thu muc results/
# python3 analysis_significance_editdist.py > analysis_significance_editdist.txt
import math
def load(cfg, yr):
    d={}
    for line in open(f'{cfg}_{yr}_results.txt', encoding='utf-8'):
        p=line.rstrip('\n').split('\t')
        if len(p)==4 and p[1] in ('CORRECT','WRONG'): d[p[0]]=(p[1]=='CORRECT', p[2], p[3])
    return d
cfgs=['abl_offline','abl_online','abl_concat','abl_cascaded','abl_dual_shared','abl_uni']
yrs=['2014','2016','2019']
D={c:{y:load(c,y) for y in yrs} for c in cfgs}
print('== SANITY ==')
for c in cfgs:
    rates=[100*sum(v[0] for v in D[c][y].values())/len(D[c][y]) for y in yrs]
    micro=100*sum(sum(v[0] for v in D[c][y].values()) for y in yrs)/3330
    print(c, ' '.join(f'{r:.2f}' for r in rates), f'micro={micro:.2f}')
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return 100*(c-h),100*(c+h)
print('\n== WILSON 95% CI (micro) ==')
for c in cfgs:
    k=sum(sum(v[0] for v in D[c][y].values()) for y in yrs)
    lo,hi=wilson(k,3330); print(f'{c:16s} {100*k/3330:6.2f}%  [{lo:.2f}, {hi:.2f}]')
def mcnemar(c1,c2):
    b=c=0
    for y in yrs:
        for k,v in D[c1][y].items():
            w=D[c2][y][k]
            if v[0] and not w[0]: b+=1
            elif w[0] and not v[0]: c+=1
    n=b+c; m=min(b,c)
    return b,c,min(2*sum(math.comb(n,i) for i in range(m+1))/2**n,1.0) if n else (b,c,1.0)
print('\n== McNEMAR exact 2-sided (pooled n=3330) ==')
for a,bb in [('abl_dual_shared','abl_concat'),('abl_dual_shared','abl_cascaded'),('abl_concat','abl_cascaded'),('abl_dual_shared','abl_online'),('abl_dual_shared','abl_uni')]:
    x,y_,p=mcnemar(a,bb); print(f'{a} vs {bb}: b={x} c={y_} p={p:.3e}')
def lev(a,b):
    if a==b: return 0
    m,n=len(a),len(b); dp=list(range(n+1))
    for i in range(1,m+1):
        prev=dp[0]; dp[0]=i
        for j in range(1,n+1):
            cur=dp[j]; dp[j]=min(dp[j]+1,dp[j-1]+1,prev+(a[i-1]!=b[j-1])); prev=cur
    return dp[n]
print('\n== ExpRate <=1/<=2 (abl_dual_shared) ==')
T=[0,0,0]
for y in yrs:
    n=len(D['abl_dual_shared'][y]); c=[0,0,0]
    for k,(ok,pr,gt) in D['abl_dual_shared'][y].items():
        d=0 if ok else lev(pr.split(),gt.split())
        for i,t in enumerate((0,1,2)):
            if d<=t: c[i]+=1
    print(f'{y}: {100*c[0]/n:.2f} | <=1 {100*c[1]/n:.2f} | <=2 {100*c[2]/n:.2f}')
    for i in range(3): T[i]+=c[i]
print(f'micro: {100*T[0]/3330:.2f} | <=1 {100*T[1]/3330:.2f} | <=2 {100*T[2]/3330:.2f}')
STRUCT=set('{ } ^ _'.split())|{'\\frac','\\sqrt','\\begin{matrix}','\\end{matrix}','\\begin{array}','\\end{array}'}
def ops(a,b):
    m,n=len(a),len(b); dp=[[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0]=i
    for j in range(n+1): dp[0][j]=j
    for i in range(1,m+1):
        for j in range(1,n+1): dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(a[i-1]!=b[j-1]))
    o=[]; i,j=m,n
    while i>0 or j>0:
        if i>0 and j>0 and dp[i][j]==dp[i-1][j-1]+(a[i-1]!=b[j-1]):
            if a[i-1]!=b[j-1]: o.append(('sub',a[i-1],b[j-1]))
            i,j=i-1,j-1
        elif i>0 and dp[i][j]==dp[i-1][j]+1: o.append(('del',a[i-1],'')); i-=1
        else: o.append(('ins','',b[j-1])); j-=1
    return o
print('\n== PHAN LOAI 860 LOI (abl_dual_shared) ==')
cls={'structural':0,'symbol':0,'insdel_other':0}; bylen={}
for y in yrs:
    for k,(ok,pr,gt) in D['abl_dual_shared'][y].items():
        L=len(gt.split()); b_=min((L-1)//10,3); t=bylen.setdefault(b_,[0,0]); t[1]+=1
        if ok: continue
        t[0]+=1; o=ops(pr.split(),gt.split())
        if any(x[1] in STRUCT or x[2] in STRUCT for x in o): cls['structural']+=1
        elif all(x[0]=='sub' and x[1] not in STRUCT and x[2] not in STRUCT for x in o): cls['symbol']+=1
        else: cls['insdel_other']+=1
print(cls)
for b_,lb in enumerate(['1-10','11-20','21-30','>30']):
    if b_ in bylen: w,n=bylen[b_]; print(f'{lb:6s}: {100*w/n:5.1f}% ({w}/{n})')
