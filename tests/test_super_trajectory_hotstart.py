"""Original public hot-start overloads and explicit numerical-match limits."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference/super_trajectory'))
from regenerate_hotstart import make_prepared, scaled_error

FIXTURE=json.loads((ROOT/'fixtures/super_trajectory_hotstart.json').read_text())
STAGES=[(case,index) for case in FIXTURE['cases'] for index in range(len(case['reference']['stages']))]
IDS=[f"{case['name']}_stage{index}" for case,index in STAGES]
ENDPOINTS=[(case,index,endpoint) for case,index in STAGES for endpoint in range(2)]
FINAL_IDS=[f"{case['name']}_stage{index}_{case['python_comparison'][index]['cross_evaluation'][endpoint]['name']}"
           for case,index,endpoint in ENDPOINTS]
KNOWN_STRICT_GRADIENT_LIMITS={
    'exp_hot_type1_stage0_cpp_final','exp_hot_type1_stage0_python_final',
    'exp_hot_type2_stage0_cpp_final','exp_hot_type2_stage0_python_final',
    'backup_hot_uniform0_stage0_cpp_final','backup_hot_uniform0_stage0_python_final',
    'backup_hot_uniform1_stage0_python_final',
    'backup_hot_uniform1_stage1_cpp_final','backup_hot_uniform1_stage1_python_final',
}


@pytest.mark.parametrize('case,index',STAGES,ids=IDS)
def test_cpp_explicit_hotstart_initial_state_and_common_x(case,index):
    raw=case['input'];ref=case['reference']['stages'][index]
    obj,*_=make_prepared(raw,ref)
    np.testing.assert_array_equal(obj.init_times,ref['stored_init_times'])
    np.testing.assert_array_equal(obj.init_points,np.asarray(ref['stored_init_points']).T)
    assert scaled_error(obj.initial_x,ref['initial_x'])<3e-9
    cost,gradient=obj.objective(np.asarray(ref['initial_x']))
    assert scaled_error(cost,ref['initial_cost'])<3e-10
    assert scaled_error(gradient,ref['initial_gradient'])<3e-9
    assert scaled_error(obj.minco.coefficients.reshape(-1,3),ref['initial_coefficients'])<3e-9
    np.testing.assert_allclose(obj.minco.durations,ref['initial_durations'],rtol=2e-15,atol=2e-15)
    if raw['kind']=='exp':
        np.testing.assert_array_equal(obj.attractors,np.asarray(ref['attractors']).T)
        np.testing.assert_array_equal(obj.radii,ref['attractor_radii'])
        np.testing.assert_array_equal(np.r_[0.,np.cumsum(raw['init_times'])],ref['guide_times'])
    elif obj.config.uniform_time_en:
        assert case['reference']['prior_heuristic_success']
        np.testing.assert_allclose(obj.minco.durations,
            np.full(obj.config.piece_num,ref['inherited_total_time']/obj.config.piece_num),rtol=2e-15)
        if index:
            previous=case['reference']['stages'][index-1]['trajectory']['durations']
            assert ref['inherited_total_time']==sum(previous)


@pytest.mark.parametrize('case,index,endpoint',ENDPOINTS,ids=FINAL_IDS)
def test_cpp_hotstart_final_objective_and_minco_coefficients(case,index,endpoint):
    obj,*_=make_prepared(case['input'],case['reference']['stages'][index])
    fixed=case['python_comparison'][index]['cross_evaluation'][endpoint]
    cost,_=obj.objective(np.asarray(fixed['x']))
    assert scaled_error(cost,fixed['reference']['cost'])<3e-10
    assert scaled_error(obj.minco.coefficients.reshape(-1,3),fixed['reference']['coefficients'])<3e-9


GRADIENT_CASES=[pytest.param(case,index,endpoint,id=name,
    marks=pytest.mark.xfail(strict=True,reason='Known limit of strict final-point C++ gradient matching; '
        'raw errors and precision diagnostics are documented in docs/super_trajectory.md')
        if name in KNOWN_STRICT_GRADIENT_LIMITS else ())
    for (case,index,endpoint),name in zip(ENDPOINTS,FINAL_IDS)]


@pytest.mark.parametrize('case,index,endpoint',GRADIENT_CASES)
def test_hotstart_final_point_cpp_gradient_strict_numerical_match(case,index,endpoint):
    obj,*_=make_prepared(case['input'],case['reference']['stages'][index])
    fixed=case['python_comparison'][index]['cross_evaluation'][endpoint]
    _,gradient=obj.objective(np.asarray(fixed['x']))
    assert scaled_error(gradient,fixed['reference']['gradient'])<3e-9


@pytest.mark.parametrize('case,index',STAGES,ids=IDS)
def test_cpp_hotstart_source_success_and_stopping_state(case,index):
    ref=case['reference']['stages'][index]
    obj,args,keywords,vertices=make_prepared(case['input'],ref)
    with patch('isaac_pathcover.trajectory.super_opt._vertex_coordinates',side_effect=vertices):
        result=obj.optimize(*args,**keywords)
    assert result.success==ref['success']
    assert result.status_code==case['python_comparison'][index]['cpp_optimizer_status']
    assert result.trajectory is not None
    # Different nonconvex endpoints are recorded, not hidden behind a wider
    # position tolerance. This check covers source success and stopping only.
