import os
from signal import signal, SIGINT
from threading import Thread
import torch
from torch import cuda
from torch.multiprocessing import Process, set_start_method
import adapm
from common.optimizer import PSAdagrad


def init_scheduler(dummy, args):
    os.environ['DMLC_NUM_SERVER'] = str(args.world_size)
    os.environ['DMLC_ROLE'] = 'scheduler'
    os.environ['DMLC_PS_ROOT_URI'] = args.root_uri
    os.environ['DMLC_PS_ROOT_PORT'] = args.root_port
    print("running scheduler")
    adapm.scheduler(args.num_keys, args.workers_per_node)


def init_node(local_rank, args, run_worker_func):
    """Start up a node (server + multiple worker threads)"""
    os.environ['DMLC_NUM_SERVER'] = str(args.world_size)
    os.environ['DMLC_ROLE'] = 'server'
    os.environ['DMLC_PS_ROOT_URI'] = args.root_uri
    os.environ['DMLC_PS_ROOT_PORT'] = args.root_port

    adapm.setup(args.num_keys, args.workers_per_node, use_techniques=args.sys_techniques, num_channels=args.communication_channels)
    server = adapm.Server(args.lens)
    rank = server.my_rank()
    print(f"Started server with rank {rank} with {args.lens.shape} keys and {args.lens.sum()} total values.")

    # make sure all servers are set up
    server.barrier()

    threads = []
    for w in range(args.workers_per_node):
        args.local_id = local_rank * args.workers_per_node + w
        worker_id = rank * args.workers_per_node + w

        # assign training device to worker
        if args.cuda:
            local_worker_id = local_rank * args.workers_per_node + w
            if args.device_ids:
                device_id = args.device_ids[local_worker_id]
            else:
                device_id = local_worker_id % cuda.device_count()
            args.device = torch.device("cuda:" + str(device_id))
        else:
            args.device = torch.device("cpu")
        # run worker
        t = Thread(target=run_worker_func, args=(worker_id, args, adapm.Worker(w, server)))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # shutdown node
    server.shutdown()


def kill_processes(signal_received, frame):
    # Kills all started processes
    print('\nSIGINT or CTRL-C detected. Shutting down all processes and exiting..')
    for p in processes:
        p.kill()
    exit(0)


processes = []

def configure_tracker_launch (args):
    tracker_env = {'DMLC_NUM_SERVER', 'DMLC_ROLE', 'DMLC_PS_ROOT_URI', 'DMLC_PS_ROOT_PORT'}
    assert os.environ.keys() >= tracker_env, f'Missing environment variables for tracker launch. Check that {tracker_env} are set.'
    args.role = os.environ['DMLC_ROLE']
    args.root_uri = os.environ['DMLC_PS_ROOT_URI']
    args.root_port = os.environ['DMLC_PS_ROOT_PORT']
    args.world_size = int(os.environ['DMLC_NUM_SERVER'])
    if args.role == 'scheduler':
        args.scheduler = True
        args.nodes = 0
    else:
        args.scheduler = False
    return args



def start_processes(worker_func, args):

    print(args)

    # catch interrupt (to shut down processes)
    signal(SIGINT, kill_processes)

    # "spawn" required for cuda training
    set_start_method('spawn', force=True)

    # launch scheduler process
    if args.scheduler:
        p = Process(target=init_scheduler, args=(0, args))
        p.start()
        processes.append(p)

    # launch training processes
    for local_rank in range(args.nodes):
        p = Process(target=init_node, args=(local_rank, args, worker_func))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
