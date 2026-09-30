"""Tinh lai ket qua dieu kien so bo (SRT chuan) tren mau so chinh thuc CROHME 986/1147/1199.
Hai mau khong co du doan (505_em_51 > 200 token; UN19_1001_em_0 khong co SRT) tinh la SAI.
So cau dung khong doi; McNemar khong doi (hai mau sai o moi cau hinh -> khong phai cap bat dong).
Quyet dinh user 25/9/2026 (theo nhan xet phan bien muc 6)."""
full={'2014':986,'2016':1147,'2019':1199}
def ed(a,b):
    a=a.split();b=b.split();d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        p=d[:];d[0]=i
        for j,y in enumerate(b,1): d[j]=min(p[j]+1,d[j-1]+1,p[j-1]+(x!=y))
    return d[-1]
out=[]
for cfg in ['seed7_test','abl_offline','abl_online','abl_concat','abl_cascaded','abl_dual_shared','abl_uni']:
    T=[0,0,0]
    line=[cfg]
    for y in full:
        L=open(f'results/{cfg}_{y}_results.txt',encoding='utf-8').read().splitlines()
        rows=[l.split('\t') for l in L if l.count('\t')>=3 and not l.startswith('Image\t')]
        k=[ed(r[2].strip(),r[3].strip()) for r in rows]
        c=sum(v==0 for v in k); e1=sum(v<=1 for v in k); e2=sum(v<=2 for v in k)
        T=[T[0]+c,T[1]+e1,T[2]+e2]
        line.append(f"{y}: n_file={len(rows)} correct={c} ExpRate={100*c/full[y]:.2f} le1={100*e1/full[y]:.2f} le2={100*e2/full[y]:.2f} (/{full[y]})")
    line.append(f"micro(/3332): ExpRate={100*T[0]/3332:.2f} le1={100*T[1]/3332:.2f} le2={100*T[2]/3332:.2f} correct={T[0]}")
    out.append('\n  '.join(line))
open('results/rev_prelim_denominator.txt','w',encoding='utf-8').write('\n'.join(out)+'\n')
