#include <stdint.h>
constexpr unsigned FULL=0xffffffffu;
__device__ unsigned next(unsigned& state){state^=state<<13;state^=state>>17;state^=state<<5;return state;}
__device__ int select(unsigned mask,unsigned choice){choice%=__popc(mask);while(choice--)mask&=mask-1;return __ffs(mask)-1;}
extern "C" __global__ void search(unsigned* schemes,unsigned* best_schemes,unsigned* random_state,int* best_ranks,int* first_hit,int* ages,unsigned long long* stats,unsigned* trace,int steps,int offset,int walks,int cap,int probes,int generalized,int adaptive,int force_period,int target,int tracing){
 int lane=threadIdx.x&31,walk=(blockIdx.x*blockDim.x+threadIdx.x)/32;if(walk>=walks)return;
 unsigned factors[4][3];int slots=cap/32;unsigned rng=random_state[walk*32+lane];
 for(int s=0;s<4;s++)for(int axis=0;axis<3;axis++)factors[s][axis]=s<slots?schemes[((size_t)walk*cap+s*32+lane)*3+axis]:0;
 int best=best_ranks[walk],stagnation=ages[walk];unsigned long long counts[6]={0,0,0,0,0,0};
 for(int step=0;step<steps;step++){
   if(best<=target)break;
   unsigned shared=__shfl_sync(FULL,next(rng),0);int base_axis=shared%3;
   int q=adaptive&&stagnation>2048?1:probes,axis=0,slot=-1,leader=-1;unsigned group=0;
   bool force=force_period>0&&stagnation>128&&(offset+step)%force_period==0;
   for(int attempt=0;attempt<3*q&&!force;attempt++){
     axis=(base_axis+attempt/q)%3;unsigned live=0;for(int s=0;s<slots;s++)if(factors[s][0])live|=1u<<s;
     slot=live?select(live,next(rng)):-1;unsigned value=slot>=0?factors[slot][axis]:0;
     unsigned matches=__match_any_sync(FULL,value);
     unsigned candidates=__ballot_sync(FULL,value&&__popc(matches)>1&&lane==__ffs(matches)-1);
     shared=__shfl_sync(FULL,next(rng),0);
     if(candidates){leader=select(candidates,shared);group=__shfl_sync(FULL,matches,leader);break;}
   }
   int kind=0;
   if(group){
     int v=(axis+1)%3,w=(axis+2)%3;int members=__popc(group);unsigned dependency=0;
     if(generalized&&members>=3&&members<=5){
       unsigned values[5]={0,0,0,0,0},remaining=group;
       for(int j=0;j<members;j++){int member=__ffs(remaining)-1;unsigned local=slot>=0?factors[slot][v]:0;values[j]=__shfl_sync(FULL,local,member);remaining&=remaining-1;}
       unsigned sum=0;for(int j=0;j<members;j++)if(lane>>j&1)sum^=values[j];
       unsigned dependencies=__ballot_sync(FULL,lane>0&&lane<(1<<members)&&!sum);
       if(dependencies){unsigned subset=__ffs(dependencies)-1;remaining=group;for(int j=0;j<members;j++){int member=__ffs(remaining)-1;if(subset>>j&1)dependency|=1u<<member;remaining&=remaining-1;}}
     }
     if(dependency){
       int pivot=__ffs(dependency)-1;unsigned local=slot>=0?factors[slot][w]:0;unsigned pivot_value=__shfl_sync(FULL,local,pivot);
       if((dependency>>lane)&1){if(lane==pivot)for(int a=0;a<3;a++)factors[slot][a]=0;else{factors[slot][w]^=pivot_value;if(!factors[slot][w])for(int a=0;a<3;a++)factors[slot][a]=0;}}
       kind=3;
     }else{
       shared=__shfl_sync(FULL,next(rng),0);int a=select(group,shared);int b=select(group&~(1u<<a),shared>>8);
       unsigned av=__shfl_sync(FULL,slot>=0?factors[slot][v]:0,a);unsigned bw=__shfl_sync(FULL,slot>=0?factors[slot][w]:0,b);
       if(lane==a)factors[slot][w]^=bw;if(lane==b)factors[slot][v]^=av;
       if((lane==a||lane==b)&&(!factors[slot][v]||!factors[slot][w]))for(int z=0;z<3;z++)factors[slot][z]=0;
       kind=1;
     }
   }else{
     shared=__shfl_sync(FULL,next(rng),0);axis=shared%3;
     unsigned live=0,empty=0;for(int s=0;s<slots;s++){if(factors[s][0])live|=1u<<s;else empty|=1u<<s;}
     unsigned occupied=__ballot_sync(FULL,live!=0),vacant=__ballot_sync(FULL,empty!=0);
     if(__popc(occupied)>=2&&vacant){
       int a=select(occupied,shared>>2),b=select(occupied&~(1u<<a),shared>>10),hole=select(vacant,shared>>18);
       int source_slot=live?select(live,next(rng)):0;int dest_slot=empty?select(empty,next(rng)):0;
       int sb=__shfl_sync(FULL,source_slot,b),sh=__shfl_sync(FULL,dest_slot,hole);
       unsigned replacement[3];for(int z=0;z<3;z++)replacement[z]=__shfl_sync(FULL,factors[source_slot][z],b);
       unsigned pivot=__shfl_sync(FULL,factors[source_slot][axis],a);
       if(lane==b){factors[sb][axis]^=pivot;if(!factors[sb][axis])for(int z=0;z<3;z++)factors[sb][z]=0;}
       replacement[axis]=pivot;if(lane==hole)for(int z=0;z<3;z++)factors[sh][z]=replacement[z];kind=2;
     }else kind=4;
   }
   if((offset+step)%64==63){for(int s=0;s<slots;s++){int shift=__shfl_sync(FULL,next(rng),0)&31;for(int z=0;z<3;z++)factors[s][z]=__shfl_sync(FULL,factors[s][z],(lane+shift)&31);}}
   int local_rank=0;for(int s=0;s<slots;s++)local_rank+=factors[s][0]!=0;
   int rank=local_rank;for(int distance=16;distance;distance>>=1)rank+=__shfl_down_sync(FULL,rank,distance);rank=__shfl_sync(FULL,rank,0);
   if(rank<best){best=rank;stagnation=0;for(int s=0;s<slots;s++)for(int z=0;z<3;z++)best_schemes[((size_t)walk*cap+s*32+lane)*3+z]=factors[s][z];}
   else stagnation++;
   if(lane==0){counts[kind==1?0:kind==2?1:kind==3?2:4]++;counts[5]++;if(best<=target&&first_hit[walk]<0)first_hit[walk]=offset+step+1;}
   if(tracing)for(int s=0;s<slots;s++)for(int z=0;z<3;z++)trace[(((size_t)walk*steps+step)*cap+s*32+lane)*3+z]=factors[s][z];
 }
 for(int s=0;s<slots;s++)for(int z=0;z<3;z++)schemes[((size_t)walk*cap+s*32+lane)*3+z]=factors[s][z];
 random_state[walk*32+lane]=rng;if(lane==0){best_ranks[walk]=best;ages[walk]=stagnation;for(int i=0;i<6;i++)stats[(size_t)walk*6+i]+=counts[i];}
}
