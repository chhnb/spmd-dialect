// Same as hydro_swe.cpp but with pointer swap instead of transfer kernel
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>

constexpr double G=9.81, HALF_G=4.905, HM1=0.001, HM2=0.01, VMIN=0.001, C1_C=0.3, MANNING_N=0.03;
using View1Di=Kokkos::View<int*>; using View1D=Kokkos::View<double*>;
using View2Di=Kokkos::View<int**>; using View2D=Kokkos::View<double**>;

KOKKOS_INLINE_FUNCTION void QF_d(double h,double u,double v,double&F0,double&F1,double&F2,double&F3){F0=h*u;F1=F0*u;F2=F0*v;F3=HALF_G*h*h;}

KOKKOS_INLINE_FUNCTION void osher_d(double QL_h,double QL_u,double QL_v,double QR_h,double QR_u,double QR_v,double FIL_in,double H_pos,double&R0,double&R1,double&R2,double&R3){
    double CR=Kokkos::sqrt(G*QR_h);double FIR_v=QR_u-2.0*CR;double fil=FIL_in,fir=FIR_v;double UA=(fil+fir)/2.0;double CA=Kokkos::fabs((fil-fir)/4.0);double CL_v=Kokkos::sqrt(G*H_pos);R0=R1=R2=R3=0.0;
    int K2=(CA<UA)?1:(UA>=0&&UA<CA)?2:(UA>=-CA&&UA<0)?3:4;int K1=(QL_u<CL_v&&QR_u>=-CR)?1:(QL_u>=CL_v&&QR_u>=-CR)?2:(QL_u<CL_v&&QR_u<-CR)?3:4;
    auto add=[&](double h,double u,double v,double s){double f0,f1,f2,f3;QF_d(h,u,v,f0,f1,f2,f3);R0+=f0*s;R1+=f1*s;R2+=f2*s;R3+=f3*s;};
    auto qs1=[&](double s){add(QL_h,QL_u,QL_v,s);};auto qs2=[&](double s){double U=fil/3,H=U*U/G;add(H,U,QL_v,s);};
    auto qs3=[&](double s){double ua=(fil+fir)/2;fil-=ua;double H=fil*fil/(4*G);add(H,ua,QL_v,s);};
    auto qs5=[&](double s){double ua=(fil+fir)/2;fir-=ua;double H=fir*fir/(4*G);add(H,ua,QR_v,s);};
    auto qs6=[&](double s){double U=fir/3,H=U*U/G;add(H,U,QR_v,s);};auto qs7=[&](double s){add(QR_h,QR_u,QR_v,s);};
    switch(K1){case 1:switch(K2){case 1:qs2(1);break;case 2:qs3(1);break;case 3:qs5(1);break;case 4:qs6(1);break;}break;
    case 2:switch(K2){case 1:qs1(1);break;case 2:qs1(1);qs2(-1);qs3(1);break;case 3:qs1(1);qs2(-1);qs5(1);break;case 4:qs1(1);qs2(-1);qs6(1);break;}break;
    case 3:switch(K2){case 1:qs2(1);qs6(-1);qs7(1);break;case 2:qs3(1);qs6(-1);qs7(1);break;case 3:qs5(1);qs6(-1);qs7(1);break;case 4:qs7(1);break;}break;
    case 4:switch(K2){case 1:qs1(1);qs6(-1);qs7(1);break;case 2:qs1(1);qs2(-1);qs3(1);qs6(-1);qs7(1);break;case 3:qs1(1);qs2(-1);qs5(1);qs6(-1);qs7(1);break;case 4:qs1(1);qs2(-1);qs7(1);break;}break;}}

struct SWEStep{int CEL;double DT;View2Di NAC,KLAS;View2D SIDE,COSF,SINF,SLCOS,SLSIN;View1D AREA,ZBC,FNC;View1D H_pre,U_pre,V_pre,Z_pre;View1D H_res,U_res,V_res,Z_res,W_res;
KOKKOS_INLINE_FUNCTION void operator()(int idx)const{
    int pos=idx+1;if(pos>CEL)return;double H1=H_pre(pos),U1=U_pre(pos),V1=V_pre(pos);double BI=ZBC(pos);
    double HI=Kokkos::fmax(H1,HM1);double UI=U1,VI=V1;if(HI<=HM2){UI=Kokkos::copysign(VMIN,UI);VI=Kokkos::copysign(VMIN,VI);}
    double ZI=Kokkos::fmax(Z_pre(pos),ZBC(pos));double WH=0,WU=0,WV=0;
    for(int j=1;j<=4;++j){int NC=NAC(j,pos);int KP=KLAS(j,pos);double COSJ=COSF(j,pos),SINJ=SINF(j,pos);double SL=SIDE(j,pos),SLCA=SLCOS(j,pos),SLSA=SLSIN(j,pos);
        double QL_h=HI,QL_u=UI*COSJ+VI*SINJ,QL_v=VI*COSJ-UI*SINJ;double CL_v=Kokkos::sqrt(G*HI);double FIL_v=QL_u+2.0*CL_v;
        double HC=0,BC=0,ZC=0,UC=0,VC=0;if(NC!=0){HC=Kokkos::fmax(H_pre(NC),HM1);BC=ZBC(NC);ZC=Kokkos::fmax(ZBC(NC),Z_pre(NC));UC=U_pre(NC);VC=V_pre(NC);}
        double f0=0,f1=0,f2=0,f3=0;
        if(KP==4||KP!=0){f3=HALF_G*H1*H1;}else if(HI<=HM1&&HC<=HM1){}else if(ZI<=BC){f0=-C1_C*Kokkos::pow(HC,1.5);f1=HI*QL_u*Kokkos::fabs(QL_u);f3=HALF_G*HI*HI;}
        else if(ZC<=BI){f0=C1_C*Kokkos::pow(HI,1.5);f1=HI*Kokkos::fabs(QL_u)*QL_u;f2=HI*Kokkos::fabs(QL_u)*QL_v;}
        else if(HI<=HM2){if(ZC>ZI){double DH=Kokkos::fmax(ZC-ZBC(pos),HM1),UN=-C1_C*Kokkos::sqrt(DH);f0=DH*UN;f1=f0*UN;f2=f0*(VC*COSJ-UC*SINJ);f3=HALF_G*HI*HI;}else{f0=C1_C*Kokkos::pow(HI,1.5);f3=HALF_G*HI*HI;}}
        else if(HC<=HM2){if(ZI>ZC){double DH=Kokkos::fmax(ZI-BC,HM1),UN=C1_C*Kokkos::sqrt(DH),HC1=ZC-ZBC(pos);f0=DH*UN;f1=f0*UN;f2=f0*QL_v;f3=HALF_G*HC1*HC1;}else{f0=-C1_C*Kokkos::pow(HC,1.5);f1=HI*QL_u*QL_u;f3=HALF_G*HI*HI;}}
        else{if(pos<NC){double QR_h=Kokkos::fmax(ZC-ZBC(pos),HM1),UR=UC*COSJ+VC*SINJ;double ratio=Kokkos::fmin(HC/QR_h,1.5),QR_u=UR*ratio;if(HC<=HM2||QR_h<=HM2)QR_u=Kokkos::copysign(VMIN,UR);double QR_v=VC*COSJ-UC*SINJ;double os0,os1,os2,os3;osher_d(QL_h,QL_u,QL_v,QR_h,QR_u,QR_v,FIL_v,H_pre(pos),os0,os1,os2,os3);f0=os0;f1=os1+(1.0-ratio)*HC*UR*UR/2;f2=os2;f3=os3;}
        else{double C1=-COSJ,S1=-SINJ,L1h=H_pre(NC),L1u=U_pre(NC)*C1+V_pre(NC)*S1,L1v=V_pre(NC)*C1-U_pre(NC)*S1;double CL1=Kokkos::sqrt(G*H_pre(NC)),FIL1=L1u+2*CL1;double HC2=Kokkos::fmax(HI,HM1),ZC1=Kokkos::fmax(ZBC(pos),ZI),R1h=Kokkos::fmax(ZC1-ZBC(NC),HM1),UR1=UI*C1+VI*S1;double ratio1=Kokkos::fmin(HC2/R1h,1.5),R1u=UR1*ratio1;if(HC2<=HM2||R1h<=HM2)R1u=Kokkos::copysign(VMIN,UR1);double R1v=VI*C1-UI*S1;double mr0,mr1,mr2,mr3;osher_d(L1h,L1u,L1v,R1h,R1u,R1v,FIL1,H_pre(NC),mr0,mr1,mr2,mr3);f0=-mr0;f1=mr1+(1-ratio1)*HC2*UR1*UR1/2;f2=mr2;double ZA=Kokkos::sqrt(mr3/HALF_G)+BC,HC3=Kokkos::fmax(ZA-ZBC(pos),0.0);f3=HALF_G*HC3*HC3;}}
        double FLR1=f1+f3,FLR2=f2;WH+=SL*f0;WU+=SLCA*FLR1-SLSA*FLR2;WV+=SLSA*FLR1+SLCA*FLR2;}
    double DTA=DT/AREA(pos),H2=Kokkos::fmax(H1-DTA*WH,HM1),Z2=H2+BI,U2=0,V2=0;
    if(H2>HM1){if(H2<=HM2){U2=Kokkos::copysign(Kokkos::fmin(VMIN,Kokkos::fabs(U1)),U1);V2=Kokkos::copysign(Kokkos::fmin(VMIN,Kokkos::fabs(V1)),V1);}
    else{double WSF=G*MANNING_N*MANNING_N*Kokkos::sqrt(U1*U1+V1*V1)/Kokkos::pow(H1,0.33333);U2=(H1*U1-DTA*WU-DT*WSF*U1)/H2;V2=(H1*V1-DTA*WV-DT*WSF*V1)/H2;U2=Kokkos::copysign(Kokkos::fmin(Kokkos::fabs(U2),15.0),U2);V2=Kokkos::copysign(Kokkos::fmin(Kokkos::fabs(V2),15.0),V2);}}
    H_res(pos)=H2;U_res(pos)=U2;V_res(pos)=V2;Z_res(pos)=Z2;W_res(pos)=Kokkos::sqrt(U2*U2+V2*V2);}};

int main(int argc,char**argv){Kokkos::initialize(argc,argv);{
    int N=argc>1?atoi(argv[1]):8192,CEL=N*N,stride=CEL+1;double dx=1.0,DT=0.5*dx/(std::sqrt(G*2.0)+1e-6);
    View2Di NAC("NAC",5,stride),KLAS("KLAS",5,stride);View2D SIDE("SIDE",5,stride),COSF("COSF",5,stride),SINF("SINF",5,stride),SLCOS("SLCOS",5,stride),SLSIN("SLSIN",5,stride);
    View1D AREA("AREA",stride),ZBC("ZBC",stride),FNC("FNC",stride);
    // 双缓冲
    View1D H[2],U[2],V[2],Z[2],W[2];
    for(int b=0;b<2;b++){H[b]=View1D("H"+std::to_string(b),stride);U[b]=View1D("U"+std::to_string(b),stride);V[b]=View1D("V"+std::to_string(b),stride);Z[b]=View1D("Z"+std::to_string(b),stride);W[b]=View1D("W"+std::to_string(b),stride);}
    // Init mesh (same as original)
    auto h_NAC=Kokkos::create_mirror_view(NAC);auto h_KLAS=Kokkos::create_mirror_view(KLAS);auto h_SIDE=Kokkos::create_mirror_view(SIDE);auto h_COSF=Kokkos::create_mirror_view(COSF);auto h_SINF=Kokkos::create_mirror_view(SINF);auto h_AREA=Kokkos::create_mirror_view(AREA);auto h_FNC=Kokkos::create_mirror_view(FNC);auto h_H=Kokkos::create_mirror_view(H[0]);auto h_Z=Kokkos::create_mirror_view(Z[0]);
    double ec[]={0,0,1,0,-1},es[]={0,-1,0,1,0};auto h_SLCOS=Kokkos::create_mirror_view(SLCOS);auto h_SLSIN=Kokkos::create_mirror_view(SLSIN);
    for(int i=0;i<N;i++)for(int j=0;j<N;j++){int p=i*N+j+1;h_AREA(p)=dx*dx;h_FNC(p)=G*MANNING_N*MANNING_N;
        for(int e=1;e<=4;e++){h_SIDE(e,p)=dx;h_COSF(e,p)=ec[e];h_SINF(e,p)=es[e];h_SLCOS(e,p)=dx*ec[e];h_SLSIN(e,p)=dx*es[e];}
        if(i>0)h_NAC(1,p)=(i-1)*N+j+1;else h_KLAS(1,p)=4;if(j<N-1)h_NAC(2,p)=i*N+j+2;else h_KLAS(2,p)=4;if(i<N-1)h_NAC(3,p)=(i+1)*N+j+1;else h_KLAS(3,p)=4;if(j>0)h_NAC(4,p)=i*N+j;else h_KLAS(4,p)=4;
        h_H(p)=j<N/2?2.0:0.5;h_Z(p)=h_H(p);}
    Kokkos::deep_copy(NAC,h_NAC);Kokkos::deep_copy(KLAS,h_KLAS);Kokkos::deep_copy(SIDE,h_SIDE);Kokkos::deep_copy(COSF,h_COSF);Kokkos::deep_copy(SINF,h_SINF);Kokkos::deep_copy(SLCOS,h_SLCOS);Kokkos::deep_copy(SLSIN,h_SLSIN);Kokkos::deep_copy(AREA,h_AREA);Kokkos::deep_copy(FNC,h_FNC);Kokkos::deep_copy(H[0],h_H);Kokkos::deep_copy(Z[0],h_Z);Kokkos::fence();
    
    int steps=10;
    // === Method A: original (compute + transfer) ===
    auto run_with_transfer=[&]()->double{
        Kokkos::deep_copy(H[0],h_H);Kokkos::deep_copy(Z[0],h_Z);Kokkos::fence();
        SWEStep swe{CEL,DT,NAC,KLAS,SIDE,COSF,SINF,SLCOS,SLSIN,AREA,ZBC,FNC,H[0],U[0],V[0],Z[0],H[1],U[1],V[1],Z[1],W[1]};
        for(int w=0;w<3;w++){for(int s=0;s<steps;s++){Kokkos::parallel_for("swe",CEL,swe);
            Kokkos::parallel_for("xfer",CEL,KOKKOS_LAMBDA(int i){int p=i+1;if(p>CEL)return;H[0](p)=H[1](p);U[0](p)=U[1](p);V[0](p)=V[1](p);Z[0](p)=Z[1](p);W[0](p)=W[1](p);});}Kokkos::fence();}
        std::vector<double> t;
        for(int r=0;r<10;r++){Kokkos::fence();auto t0=std::chrono::high_resolution_clock::now();
            for(int s=0;s<steps;s++){Kokkos::parallel_for("swe",CEL,swe);
                Kokkos::parallel_for("xfer",CEL,KOKKOS_LAMBDA(int i){int p=i+1;if(p>CEL)return;H[0](p)=H[1](p);U[0](p)=U[1](p);V[0](p)=V[1](p);Z[0](p)=Z[1](p);W[0](p)=W[1](p);});}
            Kokkos::fence();t.push_back(std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count());}
        std::sort(t.begin(),t.end());return t[5];};

    // === Method B: pointer swap (no transfer) ===
    auto run_with_swap=[&]()->double{
        Kokkos::deep_copy(H[0],h_H);Kokkos::deep_copy(Z[0],h_Z);Kokkos::fence();
        int cur=0;
        for(int w=0;w<3;w++){for(int s=0;s<steps;s++){int nxt=1-cur;
            SWEStep swe{CEL,DT,NAC,KLAS,SIDE,COSF,SINF,SLCOS,SLSIN,AREA,ZBC,FNC,H[cur],U[cur],V[cur],Z[cur],H[nxt],U[nxt],V[nxt],Z[nxt],W[nxt]};
            Kokkos::parallel_for("swe",CEL,swe);cur=nxt;}Kokkos::fence();}
        std::vector<double> t;
        for(int r=0;r<10;r++){Kokkos::fence();auto t0=std::chrono::high_resolution_clock::now();
            for(int s=0;s<steps;s++){int nxt=1-cur;
                SWEStep swe{CEL,DT,NAC,KLAS,SIDE,COSF,SINF,SLCOS,SLSIN,AREA,ZBC,FNC,H[cur],U[cur],V[cur],Z[cur],H[nxt],U[nxt],V[nxt],Z[nxt],W[nxt]};
                Kokkos::parallel_for("swe",CEL,swe);cur=nxt;}
            Kokkos::fence();t.push_back(std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count());}
        std::sort(t.begin(),t.end());return t[5];};

    double a=run_with_transfer();double b=run_with_swap();
    printf("N=%d CEL=%d (10 steps)\n",N,CEL);
    printf("  With transfer: %.3f ms\n",a);
    printf("  With swap:     %.3f ms\n",b);
    printf("  Speedup:       %.2fx\n",a/b);
    printf("  Transfer cost: %.1f%%\n",(a-b)/a*100);
}Kokkos::finalize();}
