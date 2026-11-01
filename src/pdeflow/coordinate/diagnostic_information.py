"""A limited linear initial-field information check, separate from neural inference."""
import numpy as np
from .dataset import recipe


def initial_fourier_check(c,problem,rows,solver,true_parameter):
    if problem.family!='heat' or not rows or any(r['field']!='u' or r['time']!=0. or r.get('sigma',0.)!=0. for r in rows):
        return None,{'status':'not_applicable','reason':'This check requires noiseless heat-u measurements all at t=0; no coefficient inference or later-time truth used'}
    k=recipe(c)['max_wave']
    if k>8 or len(rows)>512:return None,{'status':'not_executed','reason':'Bounded initial Fourier diagnostic requires max_wave<=8 and observations<=512'}
    modes=[(i,j) for i in range(k+1) for j in range(-k,k+1) if i>0 or j>0]
    def design(xy):
        x=(np.asarray(xy)-c.origins)/c.lengths;angle=2*np.pi*x@np.array(modes).T
        return np.c_[np.ones(len(x)),np.cos(angle),np.sin(angle)]
    a=design([[r['x'],r['y']] for r in rows]);y=np.array([r['value'] for r in rows]);coef,_,rank,singular=np.linalg.lstsq(a,y,rcond=None)
    shape=problem.grid.shape;mesh=np.meshgrid(*[o+np.arange(n)*l/n for o,l,n in zip(c.origins,c.lengths,shape)],indexing='ij');xy=np.stack(mesh,-1).reshape(-1,2);pred=(design(xy)@coef).reshape(shape)
    reference=solver.frame(0.,true_parameter,c.interval,shape)
    return pred,{'status':'evaluated','method':'Least-squares real Fourier basis at t=0; minimum coefficient norm if underdetermined; declared training max_wave, no reference-dependent mode selection',
        'columns':a.shape[1],'observations':len(rows),'rank':int(rank),'full_column_rank':bool(rank==a.shape[1]),'largest_singular_value':float(singular[0]),'smallest_reported_singular_value':float(singular[-1]),
        'sensor_mse':float(np.mean((a@coef-y)**2)),'initial_field_mse_evaluation_only':float(np.mean((pred-reference)**2)),
        'limits':'Identifiability only within this finite Fourier initial-field subspace. D is unidentifiable from t=0 alone. This is an evaluation control, not the neural model or a learned accuracy claim.'}
