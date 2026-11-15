template<class T> void emit(const T& a) {
    cout << '[';
    for (int i=0;i<a.size();++i) { if(i) cout<<','; cout<<a(i); }
    cout << ']';
}
template<class T> void emit_matrix(const T& a) {
    cout << '[';
    for (int i=0;i<a.rows();++i) {
        if(i) cout<<','; cout<<'[';
        for(int j=0;j<a.cols();++j) { if(j)cout<<',';cout<<a(i,j); }
        cout<<']';
    }
    cout<<']';
}
Eigen::VectorXd read_vector(){int n;cin>>n;Eigen::VectorXd a(n);for(int i=0;i<n;++i)cin>>a(i);return a;}
Eigen::MatrixXd read_matrix(){int n,m;cin>>n>>m;Eigen::MatrixXd a(n,m);for(int i=0;i<n;++i)for(int j=0;j<m;++j)cin>>a(i,j);return a;}

template<class O> void configure(O& o,int n,int typ,int block,double eps,int res,double rho,
 const VecDf& phys,const VecDf& bounds,const VecDf& penalties,const StatePVAJ& head,const StatePVAJ& tail){
 o.piece_num=n;o.pos_constraint_type=typ;o.block_energy_cost=block;o.smooth_eps=eps;o.integral_res=res;o.rho=rho;
 o.quadrotor_flatness.reset(phys(0),phys(1),phys(2),phys(3),phys(4),phys(5));
 o.magnitudeBounds=bounds;o.penaltyWeights=penalties;o.headPVAJ=head;o.tailPVAJ=tail;
 o.minco.setConditions(head,tail,n);o.penalty_log=VecDf::Zero(8);
 o.partialGradByCoeffs=MatD3f::Zero(8*n,3);o.partialGradByTimes=VecDf::Zero(n);
}

int last_iteration=0;
int record_iteration(void*,const VecDf&,const VecDf&,const double,const double,const int k,const int){last_iteration=k;return 0;}

template<class C> void evaluate(C& solver,VecDf x,bool optimize,double accuracy){
 auto& o=solver.opt_vars; VecDf g=VecDf::Zero(x.size());
 const double cost=C::costFunctional(&o,x,g);
 cout<<"{\"cost\":"<<cost<<",\"gradient\":";emit(g);
 cout<<",\"coefficients\":";emit_matrix(o.minco.getCoeffs());
 cout<<",\"penalty_log\":";emit(o.penalty_log);
 double energy;o.minco.getEnergy(energy);MatD3f gc;VecDf gt;
 o.minco.getEnergyPartialGradByCoeffs(gc);o.minco.getEnergyPartialGradByTimes(gt);
 Mat3Df gp;StatePVAJ gh,ge;VecDf gtime;
 o.minco.propagateGradOfWayptsAndState(gc,gt,gtime,gh,gp,ge);
 cout<<",\"energy\":"<<energy<<",\"energy_grad_coefficients\":";emit_matrix(gc);
 cout<<",\"energy_grad_time\":";emit(gt);
 cout<<",\"energy_grad_points\":";emit_matrix(gp);
 cout<<",\"energy_total_grad_time\":";emit(gtime);
 cout<<",\"energy_grad_head\":";emit_matrix(gh);
 cout<<",\"energy_grad_tail\":";emit_matrix(ge);
 if(optimize){
   lbfgs::lbfgs_parameter_t p;p.mem_size=256;p.past=3;p.min_step=1e-32;p.g_epsilon=0.;p.delta=accuracy;
   o.iter_num=0;double opt_cost;int status=lbfgs::lbfgs_optimize(x,opt_cost,&C::costFunctional,nullptr,&record_iteration,&o,p);
   cout<<",\"optimizer_status\":"<<status<<",\"optimized_cost\":"<<opt_cost<<",\"optimized_x\":";emit(x);
   cout<<",\"optimizer_iterations\":"<<last_iteration<<",\"optimizer_evaluations\":"<<o.iter_num;
   C::costFunctional(&o,x,g);
   cout<<",\"optimized_coefficients\":";emit_matrix(o.minco.getCoeffs());
   Trajectory trajectory;o.minco.getTrajectory(trajectory);
   cout<<",\"optimized_durations\":";emit(trajectory.getDurations());
   YawTrajOpt yaw;Trajectory yaw_trajectory;Vec4f initial(0.2,.1,0,0),goal(-.5,0,0,0);
   yaw.optimize(initial,goal,trajectory,yaw_trajectory,3,false,true);
   cout<<",\"yaw_durations\":";emit(yaw_trajectory.getDurations());
   cout<<",\"yaw_coefficients\":[";
   for(int i=0;i<yaw_trajectory.getPieceNum();++i){if(i)cout<<',';emit_matrix(yaw_trajectory[i].getCoeffMat().rowwise().reverse().transpose());}
   cout<<']';
   cout<<",\"yaw_higher_orders\":[";
   for(int order : {5,7}) {
    if(order==7)cout<<',';
    yaw.optimize(initial,goal,trajectory,yaw_trajectory,order,false,true);
    cout<<"{\"order\":"<<order<<",\"durations\":";emit(yaw_trajectory.getDurations());
    cout<<",\"coefficients\":[";
    for(int i=0;i<yaw_trajectory.getPieceNum();++i){if(i)cout<<',';emit_matrix(yaw_trajectory[i].getCoeffMat().rowwise().reverse().transpose());}
    cout<<"]}";
   }
   cout<<']';
 }
 cout<<"}\n";
}

int main(){
 cout<<setprecision(17);
 int mode,n,typ,block,uniform,res;double eps,rho,weight_ts,t0,te,ts;
 cin>>mode>>n>>typ>>block>>uniform>>eps>>res>>rho>>weight_ts>>t0>>te>>ts;
 VecDf phys=read_vector(),bounds=read_vector(),penalties=read_vector();
 StatePVAJ head=read_matrix(),tail=read_matrix();VecDf x=read_vector();
 int nh,nv;cin>>nh;PolyhedraH hp;for(int i=0;i<nh;++i)hp.push_back(read_matrix());
 cin>>nv;PolyhedraV vp;for(int i=0;i<nv;++i)vp.push_back(read_matrix());
 Mat3Df attractors=read_matrix();VecDf radii=read_vector();
 Trajectory exp;int exp_n;cin>>exp_n;for(int i=0;i<exp_n;++i){double t;cin>>t;exp.emplace_back(t,read_matrix());}
 bool optimize;double accuracy;cin>>optimize>>accuracy;
 if(mode==0){
   ExpTrajOpt solver;auto& o=solver.opt_vars;configure(o,n,typ,block,eps,res,rho,phys,bounds,penalties,head,tail);
   o.temporalDim=n;o.spatialDim=x.size()-n;o.hPolytopes=hp;o.vPolytopes=vp;
   o.hPolyIdx=VecDi::LinSpaced(n,0,n-1);o.vPolyIdx=VecDi::LinSpaced(n-1,0,n-2);
   o.waypoint_attractor=attractors;o.waypoint_attractor_dead_d=radii;
   evaluate(solver,x,optimize,accuracy);
 }else{
   BackupTrajOpt solver;auto& o=solver.opt_vars;configure(o,n,typ,block,eps,res,rho,phys,bounds,penalties,head,tail);
   o.temporalDim=uniform?1:n;o.spatialDim=x.size()-o.temporalDim-1;
   o.uniform_time_en=uniform;o.total_time=VecDf::Zero(1);o.gradByTotalT=VecDf::Zero(1);
   o.times=VecDf::Zero(n);o.points=Mat3Df::Zero(3,n);o.gradByPoints=Mat3Df::Zero(3,n);
   o.hPolytope=hp[0];o.vPolytope=vp[0];o.exp_traj=exp;o.min_ts=t0;o.max_ts=te;o.ts=ts;o.weight_ts=weight_ts;
   evaluate(solver,x,optimize,accuracy);
 }
}
