#include "MolLoader.h"
#include "utils/Slater.h"
double h1(int i, int j)
{
    Molecule &mol = Molecule::get_instance();
    if(i%2==j%2){
        i/=2; j/=2;
        return mol.h1e[i*mol.nCasOrb+j];
    }
    else{
        return 0;
    }
}


static double g2e_antisymm(int i,int j,int k,int l)
{
    Molecule &mol = Molecule::get_instance();

    int flag1=0, flag2=0;
    if(i%2==k%2 && j%2==l%2){ flag1=1; }
    if(i%2==l%2 && j%2==k%2){ flag2=1; }
    i/=2; j/=2; k/=2; l/=2;
    return flag1*mol.get_g2e(i,k,j,l) - flag2*mol.get_g2e(i,l,j,k);

}

static double H_elem_aaaa(SlaterInt_t s1, SlaterDiff_t diff)
{
    int sign=diff_parity_int(s1, diff);
    int m = diff.first.p[0]*2;
    int n = diff.first.p[1]*2;
    int p = diff.first.h[0]*2;
    int q = diff.first.h[1]*2;
    return sign*g2e_antisymm(m, n, p, q);
}

double H_elem_bbbb(SlaterInt_t s1, SlaterDiff_t diff)
{
    int sign=diff_parity_int(s1, diff);
    int m = diff.second.p[0]*2+1;
    int n = diff.second.p[1]*2+1;
    int p = diff.second.h[0]*2+1;
    int q = diff.second.h[1]*2+1;
    return sign*g2e_antisymm(m, n, p, q);
}

double H_elem_abab(SlaterInt_t s1, SlaterDiff_t diff)
{
    int sign=diff_parity_int(s1, diff);
    int m = diff.first.p[0]*2;
    int n = diff.second.p[0]*2+1;
    int p = diff.first.h[0]*2;
    int q = diff.second.h[0]*2+1;
    return sign*g2e_antisymm(m, n, p, q);
}

double H_elem_aa(SlaterInt_t s1, SlaterDiff_t diff)
{
    Molecule &mol = Molecule::get_instance();
    int sign=diff_parity_int(s1, diff);
    double rtn = 0;

    int m=diff.first.p[0]*2, p=diff.first.h[0]*2;
    rtn+=h1(m,p);
    for(int i=0;i<2*mol.nCasOrb;i++){
        if(i%2==0){
            if(get_bit(s1.alpha,i/2)==1 && i!=m){ rtn+=g2e_antisymm(m,i,p,i); }
        }else{
            if(get_bit(s1.beta,i/2)==1 && i!=m){ rtn+=g2e_antisymm(m,i,p,i); }
        }
    }
    return sign*rtn;
}

double H_elem_bb(SlaterInt_t s1, SlaterDiff_t diff)
{
    Molecule &mol = Molecule::get_instance();
    int sign=diff_parity_int(s1, diff);
    double rtn = 0;

    int m=diff.second.p[0]*2+1, p=diff.second.h[0]*2+1;

    rtn+=h1(m,p);
    for(int i=0;i<mol.nCasOrb*2;i++){
        if(i%2==0){
            if(get_bit(s1.alpha,i/2)==1 && i!=m){ rtn+=g2e_antisymm(m,i,p,i); }
        }else{
            if(get_bit(s1.beta,i/2)==1 && i!=m){ rtn+=g2e_antisymm(m,i,p,i); }
        }
    }
    return sign*rtn;
}

double H_elem_diag(SlaterInt_t s1)
{
    double rtn = 0;
    Molecule &mol = Molecule::get_instance();


    for(int i=0;i<2*mol.nCasOrb;i++){
        if(i%2==0){
            if(get_bit(s1.alpha,i/2)==1){
                rtn+=h1(i,i);
                for(int j=0;j<i;j++){
                    if(j%2==0){
                        if(get_bit(s1.alpha,j/2)==1){ rtn+=g2e_antisymm(i,j,i,j); }
                    }else{
                        if(get_bit(s1.beta,j/2)==1){ rtn+=g2e_antisymm(i,j,i,j); }
                    }
                }
            }
        }else{
            if(get_bit(s1.beta,i/2)==1){
                rtn+=h1(i,i);
                for(int j=0;j<i;j++){
                    if(j%2==0){
                        if(get_bit(s1.alpha,j/2)==1){ rtn+=g2e_antisymm(i,j,i,j); }
                    }else{
                        if(get_bit(s1.beta,j/2)==1){ rtn+=g2e_antisymm(i,j,i,j); }
                    }
                }
            }
        }
    }

    return rtn;
}

double Mat_element_from_state_int(SlaterInt_t Psi1,SlaterInt_t Psi2)
{
    const Molecule &mol = Molecule::get_instance();
    int nSpinOrb = mol.nCasOrb * 2;
    SlaterDiff_t diff=diff_index_int(Psi1,Psi2);
    int sign=diff_parity_int(Psi1, diff);

    int nSpaOrb = nSpinOrb/2;

    double rtn = 0;

    if(diff.first.nExc+diff.second.nExc>2){return 0.0;}


    if(diff.first.nExc==2 && diff.second.nExc==0){
        return H_elem_aaaa(Psi1, diff);
    }
    if(diff.first.nExc==0 && diff.second.nExc==2){
        return H_elem_bbbb(Psi1, diff);
    }
    if(diff.first.nExc==1 && diff.second.nExc==1){
        return H_elem_abab(Psi1, diff);
    }

    if(diff.first.nExc==1 && diff.second.nExc==0){
        return H_elem_aa(Psi1, diff);
    }
    if(diff.first.nExc==0 && diff.second.nExc==1){
        return H_elem_bb(Psi1, diff);
    }

    if(diff.first.nExc==0 && diff.second.nExc==0){
        return H_elem_diag(Psi1);
    }

    assert(0);
    return 0.;
}
