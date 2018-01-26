#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "mxlib.h"
#include "usrlib.h"
//#include "neuro.h"

#define GAMMA 0.9
#define EPSILON 0.001
#define ALPHA 0.5

/* 4*4 cell only */
#define CELL16
#define UP     0
#define LEFT   1
#define DOWN   2
#define RIGHT  3

#define N_STATE    16
#define N_ACTION    4
#define N_TRAJS  1000
#define N_FEATURE  16
#define N_T        25 

#define M_REC   4089

#define TRI    0
#define BETA   1
#define GAUSS  2

typedef struct {
  double prob[N_STATE][N_ACTION][N_STATE];
} TRANS;
typedef struct {
  int ntry;
  int s[N_T];
  int a[N_T];
  int s_next[N_T];
} TRAJ;
typedef struct {
  TRAJ traj[N_TRAJS];
  int  expertPi[N_STATE];
} TRAJS;
typedef struct {
  double val[N_STATE][N_ACTION]; /* add for bayesian */
} QVAL;

/* prottype */
//struct node **IRLneuroMake(int flag,int ni,int nj,int nexp,int ntag,int nstep,int nhidden,int *pMj);
//int IRLbackward(int flag,struct node **,int ni,int nj,double **data,double **outData,int nexp,int ntag,int step,int mj,int loop,double eta,int method);
//int IRLneuro(flag,pNode,ni,nj,data,outData,nexp,ntag,step,mj,loop,eta,method);
//int IRLfoward(int flag,struct node ** pNode,int ni,double **data,int step,int mj,int loop);
//int IRLfreeNeuro(struct node **,int nstep,int mj);
int maxActExpert(int s,int nTrajs,TRAJS *pTraj);
double calcPrior(int flag,double *reword, double Rmax);
double calcPosterior(double *reward,QVAL *pQval,TRANS *pTrans,int nTrajs,TRAJS *pTrajs,double gamma);
double calcTriPrior(double R,double Rmax);
double calcBetaPrior(double R,double Rmax);
int baysian_IRL(TRANS* pTrans, int nTrajs, TRAJS *pTrajs, int n_epoch, QVAL *pQval,int *Pi,double *U,double Pos,double *reward);
int modify_reward_random(double *reward,double r_max,double r_min);
int policy_evaluation(TRANS *pTrans,int *policy,double *reward,double *U,double *newU);
/**
  価値関数
**/
int value_iteration(pTrans, reward, U)
TRANS *pTrans;
double *reward;
double *U;  /* N_STATE out */
{
    /* Solving an MDP by value iteration */
    double U1[N_STATE];
    int s,a,s1;
    double delta;
    double Rs;
    double amax,sum;
    double p;

    double limit;
    limit =  EPSILON * (1 - GAMMA) / GAMMA;
    //n_states, n_actions, _ = trans_probs.shape
    //U1 = {s: 0 for s in range(n_states)}
    for(s=0;s<N_STATE;s++) {
      U1[s]=0.0;
      U[s]=0.0;
    }
    while (1) {
      for(s=0;s<N_STATE;s++) U[s] = U1[s];
      delta = 0;
      for(s=0;s<N_STATE;s++) {
        Rs = reward[s];
        amax=-DBL_MAX;
        for(a=0;a<N_ACTION;a++) {
          sum = 0;
          for(s1 =0;s1<N_STATE;s1++) {
            p=pTrans->prob[s][a][s1];
            sum += p * U[s1];
          }
          if(amax < sum) amax = sum;
        }
        //U1[s] = Rs + gamma * max([sum([p * U[s1] for s1, p in enumerate(trans_probs[s, a, :])])
        U1[s] = Rs + GAMMA * amax;
      }      
      delta = -DBL_MAX;
      for(s=0;s<N_STATE;s++) {
        if(delta < fabs(U1[s] - U[s])) delta = fabs(U1[s] - U[s]);
      }
      if(delta < limit) break;    
    }
    return(0);
}
/**
　行動価値関数
**/
double expected_utility(a, s, U, pTrans)
int a;
int s;
double *U;
TRANS *pTrans;
{
    /* The expected utility of doing a in state s, according to the MDP and U. */
    int s1;
    double sum;
    double p;
    //return sum([p * U[s1] for s1, p in enumerate(trans_probs[s, a, :])])
    sum = 0;
    for(s1 =0;s1<N_STATE;s1++) {
      p=pTrans->prob[s][a][s1];
      sum += p * U[s1];
    }
    return(sum);
}
/**
　最適方策
**/
int  best_policy(pTrans, U, pi)
TRANS  *pTrans;
double *U;
int    *pi; /* size state out */
{
    /*""
    Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action.
    """*/
    int s;
    int a,ma;
    double amax,v;
    double eps;
    
    for(s=0;s<N_STATE;s++) {
       amax=-DBL_MAX;
       ma = 0;
       for(a=0;a<N_ACTION;a++) {
         eps = (double)rand()/(double)RAND_MAX-0.5; /* 摂動を与える */
         v = expected_utility(a,s,U,pTrans);
         if(amax < v*(1+eps/10000.0)) {
           amax=v;
           ma = a;
         }
       }
       pi[s]=ma;
       //pi[s] = max(range(n_actions), key=lambda a: expected_utility(a, s, U, trans_probs))
       //return pi
    }
    return(0);
}
/**
  future
**/
int expected_svf(pTrans, nTrajs, pTrajs, Policy, mu)
TRANS *pTrans;
int nTrajs;
TRAJS *pTrajs;
int *Policy;
double *mu;  /* N_STATE out */
{
    //n_states, n_actions, _ = trans_probs.shape
    //n_t = len(trajs[0])
    int s,t,j,s1;
    double mut[N_STATE][N_T];
    double sum;
    //mu = np.zeros((n_states, n_t))

    for(s=0;s<N_STATE;s++) {
      for(t=0;t<N_T;t++) mut[s][t]=0;
    }
    //for traj in trajs:
    //    mu[traj[0][0], 0] += 1
    //mu[:, 0] = mu[:, 0] / len(trajs)
    for(j=0;j<nTrajs;j++) {
      mut[pTrajs->traj[j].s[0]][0] += 1;
    }
    //for(s=0;s<N_STATE;s++) mut[s][0] /= nTrajs;

    //for t in range(1, n_t):
    //    for s in range(n_states):
    //        mu[s, t] = sum([mu[pre_s, t - 1] * trans_probs[pre_s, policy[pre_s], s] for pre_s in range(n_states)])
    //return np.sum(mu, 1)
#if 0

    for(t=1;t<N_T;t++) {
      for(s=0;s<N_STATE;s++) {
        sum=0;
        for(s1=0;s1<N_STATE;s1++) {
          sum += mut[s1][t-1] * pTrans->prob[s1][Policy[s1]][s];
        }
        mut[s][t]=sum;
      }
    }
#else
    for(j=0;j<nTrajs;j++) {
      for(t=0;t<pTrajs->traj[j].ntry;t++) {
        for(s=0;s<N_STATE;s++) {
          sum=0;
          for(s1=0;s1<N_STATE;s1++) {
            sum += mut[s1][t-1] * pTrans->prob[s1][Policy[s1]][s];
          }
          mut[s][t]=sum;
        }
      }
    }
#endif
    for(s=0;s<N_STATE;s++) {
      for(sum=0,t=0;t<N_T;t++) sum += mut[s][t];
      mu[s] = sum;
    }
    for(s=0;s<N_STATE;s++) mu[s] /= nTrajs;
    return(0);
}
/**
  熟練側
***/
int featureExpert(nTrajs,pTrajs,mu)
int nTrajs;
TRAJS *pTrajs;
double *mu; /* N_STATE out */
{
   int j,s,t;

   for(s=0;s<N_STATE;s++) mu[s]=0;
   for(j=0;j<nTrajs;j++) {
#ifndef CELL16
     s = pTrajs->traj[j].s[0];
     mu[s] += 1.0;
#else
     for(t=0;t<pTrajs->traj[j].ntry;t++) {
       s = pTrajs->traj[j].s[t];
       mu[s] += 1.0;
     }   
#endif
   }
   return(0);
   
}
/**
　経路の読み込み
**/
int generate_demons(fp, /*policy,*/ pTrajs)
//int env;
FILE *fp;
//int *policy;
TRAJS *pTrajs;
{
  int j,t,k;
  char *p,record[M_REC];
  int gridEnd;
  
  //for _ in range(n_trajs):
  j=0;
#ifdef CELL16
  j=-1;
  while(fgets(record,M_REC,fp)) {
    if(record[0] == '$') continue;
    record[strlen(record)-1]='\0';
    p=strtok(record,", ");
    if(!strcmp(p,"start")) {
      if(j >= 0) { /* 前読込みのPath数 */
        pTrajs->traj[j].ntry=t;
      }
      if(j < N_TRAJS-1) j++; /* 配列超え */
      t=0;
      gridEnd=0;
      continue;
    }
    //if(gridEnd) continue;

    k=0;
    while(p) {
      if(k == 0) pTrajs->traj[j].s[t] = atoi(p);
      if(k == 1) pTrajs->traj[j].a[t] = atoi(p);
      if(k == 2) pTrajs->traj[j].s_next[t] = atoi(p);

      if(k == 2 && (atoi(p) == 0 || atoi(p) == 15)) gridEnd=1; /* 終端判断 */
      k++;
      p=strtok(NULL,", ");
    }
    if(t < N_T-1) t++; /* 配列超え */
  }
#else
  while(fgets(record,M_REC,fp)) {
    if(record[0] == '$') continue;
    record[strlen(record)-1]='\0';
    p=strtok(record,", ");
    t=0;
    k=0;
    while(p) {
      if(k == 0) pTrajs->traj[j].s[t] = atoi(p);
      if(k == 1) pTrajs->traj[j].a[t] = atoi(p);
      if(k == 2) pTrajs->traj[j].s_next[t] = atoi(p);
      k++;
      if(k == 3) {
        t++;
        k = 0;
      }
      if(t >= N_T -1) break; /* 配列超え */

      p=strtok(NULL,", ");
    }
    pTrajs->traj[j].ntry = t;
    j++;
    if(j >= N_TRAJS) break;  /* 配列超え */
  }
#endif
  return(j);
}
/************
  方策の評価
*************/
int policy_evaluation(pTrans,policy,reward,U,newU)
TRANS *pTrans;
int *policy;
double *reward;
double *U;
double *newU;
{
    int s,a;
    int s1;
    double Rs;
    double sum;
    double p;

    for(s=0;s<N_STATE;s++) {
      Rs = reward[s];
      a = policy[s];
      sum = 0;
      for(s1 =0;s1<N_STATE;s1++) {
        p=pTrans->prob[s][a][s1];
        sum += p * U[s1];
      }
      newU[s] = Rs + GAMMA * sum;
    }
    return(0);
}
/**
  遷移確率
**/
int calcTransProb(nTrajs,pTrajs,pTrans)
int nTrajs;
TRAJS *pTrajs;
TRANS *pTrans;
{
   int j,a,s,t,k,s1;

   for(s=0;s<N_STATE;s++) {
     for(a=0;a<N_ACTION;a++) {
       k=0;
       for(j=0;j<nTrajs;j++) {
         for(t=0;t<pTrajs->traj[j].ntry;t++) {
           if(pTrajs->traj[j].s[t] == s) {
             s1 = pTrajs->traj[j].s_next[t];
             if(pTrajs->traj[j].a[t] == a) pTrans->prob[s][a][s1] += 1;
           }
           k++;
         }
       }
       for(s1=0;s1<N_STATE;s1++) {
         if(k > 0) pTrans->prob[s][a][s1] /= (double)k;
         else      pTrans->prob[s][a][s1] = 0.0;
       }
     }
   }
   return(0);
}
/***
   CELL16 遷移確率
***/
int calcTransCell(pTrans)
TRANS *pTrans;
{
    int a,s,n;
    for(s=0;s<N_STATE;s++) {
      for(a=0;a<N_ACTION;a++) {
        for(n=0;n<N_STATE;n++) pTrans->prob[s][a][n]=0.0;
        if(s == 0 || s==15) pTrans->prob[s][a][s]=1.0;
        else {
          switch(a) {
          case UP   : if(s== 0 || s== 1 || s== 2 || s==3 ) pTrans->prob[s][a][s]=1;
                      else                                 pTrans->prob[s][a][s-4]=1;
                      break;
          case LEFT : if(s== 3 || s== 7 || s==11 || s==15) pTrans->prob[s][a][s]=1;
                      else                                 pTrans->prob[s][a][s+1]=1;
                      break;
          case DOWN : if(s ==12|| s==13 || s==14 || s==15) pTrans->prob[s][a][s]=1;
                      else                                 pTrans->prob[s][a][s+4]=1;
                      break;
          case RIGHT: if(s == 0|| s== 4 || s== 8 || s==12) pTrans->prob[s][a][s]=1;
                      else                                 pTrans->prob[s][a][s-1]=1;
                      break;
          }
        }
      }
    }
    return(0);
}
/************
 事後分布
*************/
double calcPosterior(reward,pQval,pTrans,nTrajs,pTrajs,gamma)
double *reward;
QVAL *pQval;
int   nTrajs;
TRAJS *pTrajs;
TRANS *pTrans;
double gamma;
{
   int s,a;
   double z,e;

   e = 0;
   for(s=0;s<N_STATE;s++) {
     /* partial function */
     for(a=0,z=0;a<N_ACTION;a++) {
       z += exp(gamma * pQval->val[s][a]);
     }
     z = log(z);
#if 0
     a = maxActExpert(s,nTrajs,pTrajs);
#else
     a = pTrajs->expertPi[s];
#endif
     e += gamma * pQval->val[s][a] - z;
   }
   e *= calcPrior(BETA,reward,10);
   return(e);
}
/*************
 事前分布
**************/
double calcPrior(flag,reward,Rmax)
int flag; /* TRI BETA GAUS */
double *reward;
double Rmax;
{
   double sum;
   int s;
   //int a;
   for(s=0,sum=0;s<N_STATE;s++) {
     switch(flag) {
     default  :
     case TRI : sum += calcTriPrior(reward[s],Rmax);
                break;
     case BETA: sum += calcBetaPrior(reward[s],Rmax);
                break;
     }
   }
   sum /= 10;
   return(sum);
}
double calcTriPrior(R,Rmax)
double R;
double Rmax;
{
   double Rmin,ans;
   
   R = fabs(R) + 0.00001;
   Rmax = Rmax + 0.00001;
   Rmin = -Rmax;

   ans = 0.4 * exp(-0.1 * pow((R - Rmax),2)) + 
         0.4 * exp(-0.1 * pow((R - Rmin),2)) + 
               exp(-0.1 * pow(R,2));

   return(ans);
}
double calcBetaPrior(R,Rmax)
double R;
double Rmax;
{
    double ans;
    R = fabs(R) + 0.00001;
    Rmax += 0.000001;
    ans = 1.0 / (pow((R / Rmax),0.5) * pow((1.0 - R / Rmax),0.5));

    return(ans);
}
/**************
  熟練者の最頻行動
***************/
int maxActExpert(s,nTrajs,pTrajs)
int s;
int nTrajs;
TRAJS *pTrajs;
{
    int t;
    int act[N_ACTION];
    int a,ma;
    int n;

    for(a=0;a<N_ACTION;a++) act[a]=0;
    for(t=0;t<nTrajs;t++) {
      for(n=0;n<pTrajs->traj[t].ntry;n++) {
        if(pTrajs->traj[t].s[n] == s) a = pTrajs->traj[t].a[n];
        act[a] ++;
      }
    }
    ma = comMaxIarray(4,act);
    return(ma);
}
/**************
  全行動価値
**************/
int allQvalue(U,pTrans,pQval)
double *U;
TRANS *pTrans;
QVAL  *pQval;
{
    int s,a;

    for(s=0;s<N_STATE;s++) {
      for(a=0;a<N_ACTION;a++) pQval->val[s][a]=expected_utility(a, s, U, pTrans);
    }
    return(0);
}
/****************
  報酬の初期化 r_max r_minの範囲
*****************/
int modify_reward_random(reward,r_max,r_min)
double *reward;
double r_max,r_min;
{
   int s;
   double width;
   double rnd;

   width = fabs(r_max - r_min);
   for(s=0;s<N_STATE;s++) {
     rnd=(double)rand()/(double)RAND_MAX;
     rnd *= width;
     rnd += r_min;
     reward[s]=rnd;
   }
   return(0);

}
/****************
  IRL Bayesian
*****************/
int baysian_IRL(pTrans, nTrajs, pTrajs, n_epoch, pQval,Pi,U,Pos,reward)
TRANS *pTrans;
int nTrajs;
TRAJS *pTrajs;
int n_epoch;
QVAL *pQval;
int    *Pi;
double *U;
double Pos;
double *reward; /* N_FEATURE out */
{
    int L;
    int s;
    //int a;

    double newU[N_STATE];
    int    newPi[N_STATE];
    double newR[N_STATE];
    double newPos;


    //value_iteration(pTrans,reward,newU);
    //best_policy(pTrans, newU, newPi);

    for(L=0;L<n_epoch;L++) {
      modify_reward_random(newR,10,-10);
#if 0
      value_iteration(pTrans,newR,newU);
#else
      policy_evaluation(pTrans,Pi,newR,U,newU);   
#endif
      best_policy(pTrans, newU, newPi);
      
      for(s=0;s<N_STATE;s++) {
        if(Pi[s] != newPi[s]) break;
      }
      if(s < N_STATE) { /* 方策が相違 */
        value_iteration(pTrans,newR,newU);  
        //best_policy(pTrans, newU, newPi);        
        /* 行動価値行列　*/
        allQvalue(newU,pTrans,pQval);
        /* 事後分布 */
        newPos=calcPosterior(newR,pQval,pTrans,nTrajs,pTrajs,GAMMA);
        /* MCMC */
        //fprintf(stderr,"new=%lf old=%lf \n",exp(newPos),exp(Pos));
        if(exp(newPos - Pos) > 1.0) {
          for(s=0;s <N_STATE;s++) {
            U[s] = newU[s];
            Pi[s] = newPi[s];
            reward[s] = newR[s];
          }
          fprintf(stderr,"AL=%4d,Pos=%lf newPos=%lf\n",L,Pos,newPos);
          Pos = newPos;
        }
      }
      else {
        /* 行動価値行列　*/
        allQvalue(newU,pTrans,pQval);
        /* 事後分布 */
        newPos=calcPosterior(newR,pQval,pTrans,nTrajs,pTrajs,GAMMA);
        /* MCMC */
        if(exp(newPos - Pos) > 1.0) {
          for(s=0;s <N_STATE;s++) {
            U[s] = newU[s];
            reward[s] = newR[s];
          }
          fprintf(stderr,"BL=%4d,Pos=%lf newPos=%lf\n",L,Pos,newPos);
          Pos = newPos;
        }
      }
    }
    return(0);

}
/**
if __name__ == '__main__':
**/
int main(argc,argv)
int argc;
char *argv[];
{
    //from envs import gridworld
 
    double reward[N_STATE];
    double U[N_STATE];
    int    pi[N_STATE];
    TRANS trans;
    TRAJS trajs;
    //double res[N_STATE];
    //double **feature_matrix;
    int s,a;
    int i,j;
    int nTrajs;
    FILE *fp,*fw;

    QVAL qval;
    double Pos;

    //grid = gridworld.GridworldEnv()
    //trans_probs, reward = trans_mat(grid)
    if(argc < 3) {
      fprintf(stderr,"USAGE: trajeryFile outptuFile\n");
      return(-9);
    }
    if(!(fp=fopen(argv[1],"r"))) {
      fprintf(stderr,"cannot read trajery file=[%s]\n",argv[1]);
      return(-1);
    }
    if(!(fw=fopen(argv[2],"w"))) {
      fprintf(stderr,"cannot write output file=[%s]\n",argv[2]);
      return(-1);
    }


    for(s=0;s<N_STATE;s++) {
      if(s == 0)         reward[s]=0.0;
      else
      if(s == N_STATE-1) reward[s]=0.0;
      else               reward[s]=-1.0;
    }
    //U = value_iteration(trans_probs, reward)
    nTrajs= generate_demons(fp, /*pi,*/ &trajs);
    fclose(fp);

#ifdef CELL16
    calcTransCell(&trans);  
#else
    calcTransProb(nTrajs,&trajs,&trans);
#endif

#if 0
    /* 価値関数 */
    value_iteration(&trans,reward,U);
    /* 最大方策 */
    best_policy(&trans, U, trajs.expertPi);
#else
    /* 熟練者の方策(事前知識) */
    for(s=0;s<N_STATE;s++) {
      trajs.expertPi[s]=maxActExpert(s,nTrajs,&trajs);
      //fprintf(stderr,"expart[%2d]=%d\n",s,trajs.expertPi[s]);
    }
#endif

    /* 報酬の初期化 */
    modify_reward_random(reward,10,-10);
    /* 価値関数 */
    value_iteration(&trans,reward,U);
    /* 最大方策 */
    best_policy(&trans, U, pi);
    /* 行動価値行列　*/
    for(s=0;s<N_STATE;s++) {
      for(a=0;a<N_ACTION;a++) qval.val[s][a]=expected_utility(a, s, U, &trans);
    }


    /* 事後分布 */
    Pos=calcPosterior(reward,&qval,&trans,nTrajs,&trajs,0.95);  /* gamma=0.95 */


    /* ベイジアンIRL */
    baysian_IRL(&trans, nTrajs, &trajs, 3000, &qval, pi, U, Pos,reward);

#ifdef CELL16
    for(s=0;s<N_STATE;s++) {
      fprintf(stderr,"reword[%2d]=%lf\n",s,reward[s]);
    }
    for(i=0;i<4;i++) {
      for(j=0;j<4;j++) {
        s = i * 4 + j;
        fprintf(fw,"%lf",reward[s]);
        if(j < 4-1) fprintf(fw,",");
        else        fprintf(fw,"\n");
      }
    }
#else
    for(s=0;s<N_STATE;s++) {
      fprintf(stderr,"reword[%2d]=%lf\n",s,reward[s]);
      fprintf(fw,"%2d,%lf\n",s,reward[s]);
    }
#endif

    /* import matplotlib.pyplot as plt
    def to_mat(res, shape):
        dst = np.zeros(shape)
        for i, v in enumerate(res):
            dst[i / shape[1], i % shape[1]] = v
        return dst

    plt.matshow(to_mat(res, grid.shape))
    plt.show()
    **/
    fclose(fw);

    /* free */
    //for(i=0;i<N_FEATURE;i++) free(feature_matrix[i]);
    //free(feature_matrix);

    return(0);
}