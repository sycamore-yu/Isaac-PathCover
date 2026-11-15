"""Fixed C++ input/output checks for the SUPER trajectory pipeline."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from numba import njit
from isaac_pathcover.trajectory.lbfgs import minimize, LBFGSParameters
from isaac_pathcover.trajectory.numerics import flatness_jacobian
from isaac_pathcover.trajectory.minco import PiecewisePolynomial
from isaac_pathcover.trajectory.super_opt import YawTrajOpt

ROOT=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('super_oracle',ROOT/'reference/super_trajectory/regenerate.py')
ORACLE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(ORACLE)
CASES=json.loads((ROOT/'fixtures/super_trajectory_reference.json').read_text())['cases']


def scaled_error(actual,expected):
    expected=np.asarray(expected)
    return np.max(np.abs(np.asarray(actual)-expected)/(1+np.abs(expected)),initial=0.)


@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_cpp_objective_gradient_and_minco_state_adjoint(case):
    obj,x=ORACLE.deserialize(case['input']);ref=case['reference'];cost,gradient=obj.objective(x)
    assert scaled_error(cost,ref['cost'])<3e-10
    assert scaled_error(gradient,ref['gradient'])<3e-9
    assert scaled_error(obj.penalty_log,ref['penalty_log'])<3e-9
    assert scaled_error(obj.minco.coefficients.reshape(-1,3),ref['coefficients'])<3e-9
    energy,gc,gt=obj.minco.energy_with_partials()
    gp,total_t,gh,ge=obj.minco.propagate_gradient(gc,gt,boundary_gradients=True)
    assert scaled_error(energy,ref['energy'])<3e-9
    assert scaled_error(gc.reshape(-1,3),ref['energy_grad_coefficients'])<3e-9
    assert scaled_error(gt,ref['energy_grad_time'])<3e-9
    assert scaled_error(gp.T,ref['energy_grad_points'])<3e-9
    assert scaled_error(total_t,ref['energy_total_grad_time'])<3e-9
    assert scaled_error(gh,ref['energy_grad_head'])<3e-9
    assert scaled_error(ge,ref['energy_grad_tail'])<3e-9


@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
@pytest.mark.parametrize('order',[3,5,7])
def test_cpp_yaw_polynomial(case,order):
    ref=case['reference'];position=PiecewisePolynomial(ref['optimized_durations'],
        np.array(ref['optimized_coefficients']).reshape(-1,8,3))
    yaw=YawTrajOpt().optimize([.2,.1,0.,0.],[-.5,0.,0.,0.],position,order=order)
    durations=ref['yaw_durations'] if order==3 else ref['yaw_higher_orders'][(order-5)//2]['durations']
    coefficients=ref['yaw_coefficients'] if order==3 else ref['yaw_higher_orders'][(order-5)//2]['coefficients']
    np.testing.assert_allclose(yaw.durations,durations,rtol=1e-12,atol=1e-12)
    coefficients=np.asarray(coefficients)
    expected=np.zeros_like(yaw.coefficients)
    expected[:,:coefficients.shape[1]]=coefficients
    assert scaled_error(yaw.coefficients,expected)<3e-10


@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_cpp_lbfgs_convergence_and_trajectory(case):
    obj,x=ORACLE.deserialize(case['input']);ref=case['reference'];result=obj.solve_prepared(x)
    assert result.status==ref['optimizer_status']
    obj.objective(result.x)
    actual=obj.minco.trajectory()
    expected=PiecewisePolynomial(ref['optimized_durations'],np.array(ref['optimized_coefficients']).reshape(-1,8,3))
    # Nonconvex solves may stop at adjacent iterates under the original tolerance.
    assert scaled_error(result.cost,ref['optimized_cost'])<1e-3
    times=np.linspace(0.,min(actual.duration,expected.duration),101)
    assert np.max(np.linalg.norm(actual.evaluate(times)-expected.evaluate(times),axis=1))<.02


def test_exact_polynomial_prefix_and_suffix_join():
    ref=CASES[0]['reference'];trajectory=PiecewisePolynomial(ref['optimized_durations'],
        np.array(ref['optimized_coefficients']).reshape(-1,8,3))
    split=.37*trajectory.duration
    prefix=trajectory.cut(0.,split);suffix=trajectory.cut(split,trajectory.duration)
    joined=prefix.concatenate(suffix)
    times=np.linspace(0.,trajectory.duration,101)
    for d in range(4):
        np.testing.assert_allclose(joined.evaluate(times,d),trajectory.evaluate(times,d),rtol=2e-9,atol=2e-9)
        np.testing.assert_allclose(prefix.evaluate(split,d),suffix.evaluate(0.,d),rtol=2e-9,atol=2e-9)


def test_yaw_half_turn_preserves_source_side():
    assert YawTrajOpt._normalize(0.,np.pi)==np.pi
    assert YawTrajOpt._normalize(0.,-np.pi)==-np.pi
    assert YawTrajOpt._normalize(.4,float('nan'))==.4


@njit(cache=True)
def quadratic_objective(x,args):
    return .5*np.dot(x,x),x.copy()


def test_lbfgs_initial_stationary_and_step_bound():
    stationary=minimize(quadratic_objective,np.zeros(2),LBFGSParameters(g_epsilon=0.))
    assert (stationary.status,stationary.iterations,stationary.evaluations)==(0,0,1)
    bounded=minimize(quadratic_objective,np.array([.1]),LBFGSParameters(max_step=1.,max_iterations=1))
    assert bounded.status==-1008
    np.testing.assert_allclose(bounded.x,[.05],atol=1e-15,rtol=0.)


def test_runtime_flatness_jacobian_matches_cpp_nonzero_drag_yaw():
    expected=json.loads((ROOT/'fixtures/trajectory_reference.json').read_text())['expected']['flatness']
    thrust,quat,omega,tj,qj,oj=flatness_jacobian(np.array([.8,-.3,.2]),np.array([.5,.2,.8]),
        np.array([.2,.7,-.4]),.3,-.2,np.array([.752,9.8066,.1,.15,.08,.001]))
    gradient=.7*tj+np.array([.3,-.6,.2,.1]) @ qj+np.array([.4,-.2,.5]) @ oj
    gradient[:3]+=np.array([.1,.3,-.2])
    values=[thrust,quat,omega,np.array([.2,-.1,.4]),gradient[:3],gradient[3:6],gradient[6:9],gradient[9],gradient[10]]
    names=['thrust','quat','omega','grad_pos','grad_vel','grad_acc','grad_jer','grad_yaw','grad_yaw_rate']
    for name,value in zip(names,values):
        np.testing.assert_allclose(value,expected[name],rtol=2e-14,atol=2e-15)
