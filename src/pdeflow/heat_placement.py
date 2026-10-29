"""Reference-assisted synthetic sensor design; never an inference feature.

Heat focus balances retained contrast extrema, reserves uniform background
samples, and uses broad wrapped Gaussian jitter. This is not blind active sensing.
"""
import numpy as np


def evenly_spaced(n,lengths=(1.,1.),origins=(0.,0.)):
    if type(n)!=int or not 1<=n<=20000:raise ValueError('Sensor count must be 1..20000')
    lengths=np.asarray(lengths,float);origins=np.asarray(origins,float)
    rows=max(1,min(n,int(round(np.sqrt(n*lengths[1]/lengths[0])))))
    base,extra=divmod(n,rows);points=[];covered=0
    for j in range(rows):
        count=base+((j+1)*extra//rows-j*extra//rows)
        points.extend([((i+.5)/count,(covered+count/2)/n) for i in range(count)]);covered+=count
    return np.asarray(points)*lengths+origins


def features(field,lengths,origins):
    u=np.asarray(field,float);contrast=u-u.mean();scale=float(np.max(np.abs(contrast)))
    if scale<=1e-8*max(1,float(np.max(np.abs(u)))):return [],'Negligible contrast: using uniform coverage.'
    # Smooth on a fixed fraction of physical min-domain length, periodic FFT.
    wave=np.meshgrid(*[2*np.pi*np.fft.fftfreq(n,d=L/n) for n,L in zip(u.shape,lengths)],indexing='ij')
    smooth=np.fft.ifftn(np.fft.fftn(contrast)*np.exp(-.5*(.025*min(lengths))**2*sum(k*k for k in wave))).real
    score=np.abs(smooth);maximum=score.max();local=np.ones(u.shape,bool)
    for dx in (-1,0,1):
        for dy in (-1,0,1):local&=score>=np.roll(score,(dx,dy),(0,1))-1e-14
    mask=local & (score>=.12*maximum);visited=np.zeros_like(mask);candidates=[]
    for seed in np.argwhere(mask):
        if visited[tuple(seed)]:continue
        stack=[tuple(seed)];visited[tuple(seed)]=True;component=[]
        while stack:
            a=stack.pop();component.append(a)
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    b=((a[0]+dx)%u.shape[0],(a[1]+dy)%u.shape[1])
                    if mask[b] and not visited[b]:visited[b]=True;stack.append(b)
        candidates.append(min(component,key=lambda ij:(-score[ij],*ij)))
    candidates=sorted(candidates,key=lambda ij:(-score[tuple(ij)],*ij))
    centres=[]
    for ij in candidates:
        p=np.asarray(origins)+ij/np.asarray(u.shape)*lengths
        if all(np.linalg.norm(np.minimum(np.abs(p-c),np.asarray(lengths)-np.abs(p-c)))>=.18*min(lengths) for c in centres):centres.append(p)
        if len(centres)==8:break
    return [c.tolist() for c in centres],'' if centres else 'No resolved contrast extrema: using uniform coverage.'


def place(n,policy='random',seed=91,lengths=(1.,1.),origins=(0.,0.),field=None,centres=None,background=.25,spread=.08,minimum_distance=.005,manual=None):
    if type(n)!=int or not 1<=n<=20000:raise ValueError('Sensor count must be 1..20000')
    L=np.asarray(lengths,float);o=np.asarray(origins,float)
    if L.shape!=(2,) or o.shape!=(2,) or not np.isfinite([*L,*o]).all() or min(L)<=0:raise ValueError('Positive finite 2D domain required')
    if not np.isfinite([background,spread,minimum_distance]).all() or not .1<=background<=1 or not .04<=spread<=.4 or not 0<=minimum_distance<=.1:raise ValueError('Focus settings: background .1..1, spread .04...4, minimum distance0...1 (fractions of minimum domain length)')
    provided=centres is not None
    rng=np.random.default_rng(seed);message='';allocation=[];centres=centres or [];background_count=n;relaxations=0
    if policy=='manual':
        xy=np.asarray(manual,float)
        if xy.shape!=(n,2):raise ValueError('Manual count must equal the number of sensor points')
    elif policy=='even':xy=evenly_spaced(n,L,o)
    elif policy in ('random','heat'):
        xy=o+rng.random((n,2))*L
        if policy=='heat':
            if field is None and not centres:raise ValueError('Heat-focused placement needs a reference field or explicit ROI centres; unavailable in measurements-only mode')
            if not centres and not provided:centres,message=features(field,L,o)
            elif not centres:message='Negligible/no retained contrast regions: using uniform coverage.'
            if centres:
                background_count=min(n,max(1,int(np.ceil(background*n))));focused=n-background_count
                order=rng.permutation(len(centres));counts=np.full(len(centres),focused//len(centres));counts[order[:focused%len(centres)]]+=1;allocation=counts.tolist()
                cursor=background_count
                for centre,count in zip(centres,counts):
                    for local_index in range(int(count)):
                        for attempt in range(12):
                            jitter=rng.normal(size=2)*spread*min(L)
                            if local_index==0:jitter=np.clip(jitter*.25,-.5*spread*min(L),.5*spread*min(L))
                            point=o+(np.asarray(centre)-o+jitter)%L
                            if cursor==0:break
                            delta=np.abs(xy[:cursor]-point);distance=np.linalg.norm(np.minimum(delta,L-delta),axis=1)
                            if np.min(distance)>=minimum_distance*min(L):break
                        else:relaxations+=1 # bounded relaxation, no infinite rejection loop
                        xy[cursor]=point;cursor+=1
                if focused<len(centres):message='Focused budget is smaller than retained regions; a seeded balanced subset is covered.'
    else:raise ValueError('Placement must be random, even, heat or manual')
    if not np.isfinite(xy).all() or np.any(xy<o) or np.any(xy>=o+L) or len(set(map(tuple,xy)))!=n:raise ValueError('Sensors must be unique inside the endpoint-excluded periodic domain')
    return xy,{'policy':policy,'seed':int(seed),'reference_assisted':policy=='heat','background_fraction':background,'background_count':background_count,
       'spread_fraction_min_length':spread,'minimum_distance_fraction_min_length':minimum_distance,'distance_applies_to':'Focused samples against prior points; uniform background is not repelled','distance_relaxations':relaxations,'retained_regions':centres,'allocation':allocation,
       'detector':{'smoothing_fraction_min_length':.025,'separation_fraction_min_length':.18,'maximum_regions':8,'contrast_threshold_fraction':.12},'analysis_shape':list(np.shape(field)) if field is not None else None,'message':message,
       'strategy':'Equal stratified region allocation plus uniform background; broad periodic Gaussian kernels' if policy=='heat' else 'Balanced row lattice with cell centres (approximately even for non-square counts)' if policy=='even' else policy,
       'selection_bias':'Positions selected from reference contrast convey information; not statistically equivalent to random sensing' if policy=='heat' else None}
