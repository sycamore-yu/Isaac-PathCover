// The class methods called here are extracted unchanged from SUPER 2ad3419.
// The objective wrapper records its first evaluation and forwards every call.
void emit_trajectory(const Trajectory& trajectory) {
 cout<<"{\"durations\":";emit(trajectory.getDurations());cout<<",\"coefficients\":[";
 for(int i=0;i<trajectory.getPieceNum();++i) {
  if(i)cout<<',';
  emit_matrix(trajectory[i].getCoeffMat().rowwise().reverse().transpose());
 }
 cout<<"]}";
}

void start_capture() {captured_x.resize(0);last_x.resize(0);capture_enabled=true;}
void emit_points(const vec_Vec3f& points) {
 Mat3Df matrix(3,points.size());
 for(int i=0;i<points.size();++i)matrix.col(i)=points[i];
 emit_matrix(matrix);
}
void emit_capture(bool success,const Trajectory& trajectory) {
 cout<<"{\"success\":"<<(success?"true":"false")<<",\"initial_x\":";emit(captured_x);
 cout<<",\"initial_cost\":"<<captured_cost<<",\"initial_gradient\":";emit(captured_gradient);
 cout<<",\"initial_coefficients\":";emit_matrix(captured_coefficients);
 cout<<",\"initial_durations\":";emit(captured_times);
 cout<<",\"last_x\":";emit(last_x);cout<<",\"last_cost\":"<<last_cost;
 cout<<",\"trajectory\":";emit_trajectory(trajectory);
}

template<class C> void configure_class(C& solver,int n,int typ,int block,double eps,int res,
 double rho,double accuracy,const VecDf& phys,const VecDf& bounds,const VecDf& penalties,
 const StatePVAJ& head,const StatePVAJ& tail) {
 configure(solver.opt_vars,n,typ,block,eps,res,rho,phys,bounds,penalties,head,tail);
 auto& c=solver.cfg_;c.piece_num=n;c.pos_constraint_type=typ;c.opt_accuracy=accuracy;
 c.max_vel=bounds(0);c.max_acc=bounds(1);c.max_omg=bounds(3);
 c.penna_pos=penalties(0);c.penna_vel=penalties(1);c.penna_acc=penalties(2);
 c.penna_omg=penalties(5);c.penna_thr=penalties(6);
}

int main() {
 cout<<setprecision(17);
 int mode,n,typ,block,uniform,res,repeat;double eps,rho,weight_ts,t0,te,ts,accuracy,prior_duration;
 cin>>mode>>n>>typ>>block>>uniform>>eps>>res>>rho>>weight_ts>>t0>>te>>ts>>accuracy>>prior_duration>>repeat;
 VecDf phys=read_vector(),bounds=read_vector(),penalties=read_vector();
 StatePVAJ head=read_matrix(),tail=read_matrix();
 VecDf init_times=read_vector();Mat3Df init_matrix=read_matrix();vec_Vec3f init_points;
 for(int i=0;i<init_matrix.cols();++i)init_points.emplace_back(init_matrix.col(i));
 int nh;cin>>nh;PolytopeVec polytopes;
 for(int i=0;i<nh;++i)polytopes.emplace_back(read_matrix());
 Trajectory exp;int exp_n;cin>>exp_n;
 for(int i=0;i<exp_n;++i){double t;cin>>t;exp.emplace_back(t,read_matrix());}
 Trajectory trajectory;
 std::ostringstream logs;auto* stream=cout.rdbuf();
 if(mode==0) {
  ExpTrajOpt solver;configure_class(solver,n,typ,block,eps,res,rho,accuracy,phys,bounds,penalties,head,tail);
  start_capture();cout.rdbuf(logs.rdbuf());
  const bool success=solver.optimize(head,tail,polytopes,init_points,init_times,trajectory);
  cout.rdbuf(stream);cout<<std::defaultfloat<<setprecision(17);
  cout<<"{\"stages\":[";emit_capture(success,trajectory);
  cout<<",\"stored_init_times\":";emit(solver.opt_vars.init_ts);
  cout<<",\"stored_init_points\":";emit_points(solver.opt_vars.init_ps);
  cout<<",\"attractors\":";emit_matrix(solver.opt_vars.waypoint_attractor);
  cout<<",\"attractor_radii\":";emit(solver.opt_vars.waypoint_attractor_dead_d);
  cout<<",\"guide_times\":[";
  for(int i=0;i<solver.opt_vars.guide_t.size();++i){if(i)cout<<',';cout<<solver.opt_vars.guide_t[i];}
  cout<<"],\"vertices\":[";
  for(int i=1;i<solver.opt_vars.vPolytopes.size();i+=2){if(i>1)cout<<',';emit_matrix(solver.opt_vars.vPolytopes[i].transpose());}
  cout<<"]}]}\n";
 } else {
  BackupTrajOpt solver;configure_class(solver,n,typ,block,eps,res,rho,accuracy,phys,bounds,penalties,head,tail);
  auto& o=solver.opt_vars;o.weight_ts=weight_ts;o.uniform_time_en=uniform;
  o.total_time=VecDf::Zero(1);o.gradByTotalT=VecDf::Zero(1);
  double out_ts=ts;bool prior_success=true;
  if(uniform) {
   cout.rdbuf(logs.rdbuf());VecDf end=init_points.back();
   prior_success=solver.optimize(exp,t0,te,ts,end,prior_duration,polytopes[0],trajectory,out_ts,false);
   cout.rdbuf(stream);
  }
  cout<<"{\"prior_heuristic_success\":"<<(prior_success?"true":"false")<<",\"stages\":[";
  for(int k=0;k<repeat;++k) {
   if(k)cout<<',';
   const double inherited=o.total_time(0);
   start_capture();cout.rdbuf(logs.rdbuf());
   const bool success=solver.optimize(exp,t0,te,ts,polytopes[0],init_times,init_points,trajectory,out_ts);
   cout.rdbuf(stream);cout<<std::defaultfloat<<setprecision(17);
   emit_capture(success,trajectory);
   cout<<",\"inherited_total_time\":"<<inherited<<",\"out_ts\":"<<out_ts;
   cout<<",\"stored_init_times\":";emit(o.init_t_vec);
   cout<<",\"stored_init_points\":";emit_points(o.init_ps);
   cout<<",\"vertices\":[";emit_matrix(o.vPolytope.transpose());cout<<"]}";
   capture_enabled=false;
  }
  cout<<"]}\n";
 }
}
