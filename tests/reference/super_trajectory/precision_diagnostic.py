"""Offline precision probe of the unchanged objective expressions."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from regenerate_hotstart import make_prepared
from regenerate import serialize

root=Path(__file__).resolve().parents[3]
fixture=json.loads((root/'tests/fixtures/super_trajectory_hotstart.json').read_text())
case=fixture['cases'][-1];stage=case['reference']['stages'][-1]
obj,*_=make_prepared(case['input'],stage)
x=np.asarray(stage['last_x']);cpp=case['python_comparison'][-1]['cross_evaluation'][0]['reference']
f,g=obj.objective(x)

class Precision(ast.NodeTransformer):
    def __init__(self,extended):self.extended=extended
    def visit_FunctionDef(self,node):
        node.decorator_list=[]
        return self.generic_visit(node)
    def visit_Call(self,node):
        self.generic_visit(node)
        if (self.extended and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
            and node.func.value.id=='np' and node.func.attr in ('zeros','ones','empty','array')
            and not any(k.arg=='dtype' for k in node.keywords)):
            node.keywords.append(ast.keyword(arg='dtype',value=ast.Attribute(value=ast.Name(id='np',ctx=ast.Load()),attr='longdouble',ctx=ast.Load())))
        return node

functions={}
source=(root/'src/isaac_pathcover/trajectory/numerics.py').read_text()
for mode in (False,True):
    tree=Precision(mode).visit(ast.parse(source));ast.fix_missing_locations(tree);namespace={}
    exec(compile(tree,'offline_precision_numerics.py','exec'),namespace)
    functions[mode]=namespace['objective_full']
problem_ld=tuple(a.astype(np.longdouble) if a.dtype.kind=='f' else a.copy() for a in obj.problem)
f64=functions[False](x,obj.problem)
ld=functions[True](x.astype(np.longdouble),problem_ld)
grad_cpp=np.asarray(cpp['gradient']);grad_ld=ld[1]
index=int(np.argmax(abs(g-grad_cpp)/(1+abs(grad_cpp))))
m=obj.minco;A=np.zeros((m.width*m.pieces,m.width*m.pieces))
A[m._rows,m._cols]=m._factors*m.durations[m._ti]**m._powers
A[m._fixed_rows,m._fixed_cols]=m._fixed_values
rows=[]
oracle=root/'.reference/super_trajectory/oracle'
for h in (1e-6,1e-7,1e-8,1e-9,3e-10,1e-10,3e-11,1e-11,3e-12,1e-12,3e-13,1e-13,3e-14,1e-14):
    py=[];cc=[];precise=[]
    for multiple in (-2,-1,1,2):
        point=x.copy();point[index]+=multiple*h
        py.append(obj.objective(point)[0])
        ref=json.loads(subprocess.run([oracle],input=serialize(obj,point,False),text=True,capture_output=True,check=True).stdout)
        cc.append(ref['cost'])
        point_ld=x.astype(np.longdouble);point_ld[index]+=np.longdouble(multiple)*np.longdouble(str(h))
        precise.append(functions[True](point_ld,problem_ld)[0])
    rows.append({'step':h,'python_5point':float((py[0]-8*py[1]+8*py[2]-py[3])/(12*h)),
                 'cpp_5point':float((cc[0]-8*cc[1]+8*cc[2]-cc[3])/(12*h)),
                 'longdouble_5point':float((precise[0]-8*precise[1]+8*precise[2]-precise[3])/(np.longdouble(12)*np.longdouble(str(h))))})
out={'case':case['name'],'stage':1,'input_x':x.tolist(),'index':index,
     'numerics_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
     'durations':stage['trajectory']['durations'],'minco_system_condition_2':float(np.linalg.cond(A)),
     'longdouble_mantissa_bits':np.finfo(np.longdouble).nmant,
     'cost_cpp':cpp['cost'],'cost_numba':float(f),'cost_python_interpreted':float(f64[0]),'cost_longdouble':float(ld[0]),
     'gradient_cpp':grad_cpp.tolist(),'gradient_numba':g.tolist(),
     'gradient_python_interpreted':f64[1].tolist(),'gradient_longdouble':[float(v) for v in grad_ld],
     'cpp_max_scaled_gradient_error_to_longdouble':float(np.max(abs(grad_cpp-grad_ld)/(1+abs(grad_ld)))),
     'numba_max_scaled_gradient_error_to_longdouble':float(np.max(abs(g-grad_ld)/(1+abs(grad_ld)))),
     'interpreted_float64_max_scaled_gradient_error_to_numba':float(np.max(abs(g-f64[1])/(1+abs(g)))),
     'finite_differences':rows,
     'method':'Removed Numba decorators; promoted floating array storage and inputs to numpy.longdouble. Arithmetic expressions, loop ordering, constraints and weights remain the runtime Python source. Offline diagnostic only.'}
target=root/'results/super_risp_development_20261115/profile/hotstart_precision_diagnostic.json'
target.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
(root/'tests/fixtures/super_trajectory_precision_diagnostic.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if not k.startswith('gradient_') and k!='input_x'},indent=2))
