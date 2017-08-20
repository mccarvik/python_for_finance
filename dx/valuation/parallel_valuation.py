import multiprocessing as mp


def simulate_parallel(objs, fixed_seed=True):
    procs = []
    man = mp.Manager()
    output = man.Queue()

    def worker(o, output):
        o.generate_paths(fixed_seed=fixed_seed)
        output.put((o.name, o))
    for o in objs:
        procs.append(mp.Process(target=worker, args=(o, output)))
    [pr.start() for pr in procs]
    [pr.join() for pr in procs]
    results = [output.get() for o in objs]
    underlying_objects = {}
    for o in results:
        underlying_objects[o[0]] = o[1]
    return underlying_objects


def value_parallel(objs, fixed_seed=True, full=False):
    procs = []
    man = mp.Manager()
    output = man.Queue()

    def worker(o, output):
        if full is True:
            pvs = o.present_value(fixed_seed=fixed_seed, full=True)[1]
            output.put((o.name, pvs))
        else:
            pv = o.present_value(fixed_seed=fixed_seed)
            output.put((o.name, pv))

    for o in objs:
        procs.append(mp.Process(target=worker, args=(o, output)))
    [pr.start() for pr in procs]
    [pr.join() for pr in procs]
    res_list = [output.get() for o in objs]
    results = {}
    for o in res_list:
        results[o[0]] = o[1]
    return results


def greeks_parallel(objs, Greek='Delta'):
    procs = []
    man = mp.Manager()
    output = man.Queue()

    def worker(o, output):
        if Greek == 'Delta':
            output.put((o.name, o.delta()))
        elif Greek == 'Vega':
            output.put((o.name, o.vega()))

    for o in objs:
        procs.append(mp.Process(target=worker, args=(o, output)))
    [pr.start() for pr in procs]
    [pr.join() for pr in procs]
    res_list = [output.get() for o in objs]
    results = {}
    for o in res_list:
        results[o[0]] = o[1]
    return results