import functools

def rgetattr(obj, path): # recursive getattr: eg. elmo.scalar_mix.gamma
    return functools.reduce(getattr, path.split('.'), obj)

def rsetattr(obj, path, val): # recursive setattr: eg. elmo.scalar_mix.gamma
    pre, _, post = path.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)