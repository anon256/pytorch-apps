import torch
import time
import os
import torch.nn as nn
import tqdm
import sys
import ctr.criteo_dataset
import common.process_setup
import ctr.setup
from ctr.cli import parse_arguments
from common.ps_models import PSDense
from common.optimizer import PSAdagrad
from common.iterator import PrefetchIterator
import functools
from sklearn.metrics import log_loss

def run_worker(worker_id, args, kv):
    train(worker_id, args, kv)
    kv.barrier()  # wait for all workers to finish
    kv.finalize()

def run_eval(model, test_dl, args, epoch):
    # file name of best performing permutation of this model, e.g. CriteoNetwork_best_model.pt
    best_model_path = type(model.model).__name__ + "_best_model.pt"
    loss_sum = 0.0
    loss_sum_bce = 0.0
    bce = nn.BCEWithLogitsLoss()

    with tqdm.tqdm(test_dl) as tq, torch.no_grad():
        for batch_id, data in enumerate(tq):
            model.pull()
            labels = data['labels'].to(args.device)

            pred = model(data["dense_features"].to(args.device), data["sparse_features"].to(args.device))

            loss_sum_bce += bce(input=pred, target=labels)

            # regular logloss w/ extra sigmoid beforehand
            sig_pred = torch.sigmoid(pred)
            loss = log_loss(y_true=labels.cpu().data.numpy(), y_pred=sig_pred.cpu().data.numpy().astype("float64"))
            loss_sum += loss

        avg_test_logloss = loss_sum / len(tq)
        avg_test_logloss_bce = loss_sum_bce / len(tq)
        print(f"EVAL: Epoch {epoch} average logloss {avg_test_logloss} , BCE-loss={avg_test_logloss_bce}")

        if not hasattr(args, 'best_loss'):  # piggyback off args
            args.best_loss = float('inf')
        if args.save_model and args.best_loss > avg_test_logloss:
            args.best_loss = avg_test_logloss
            #torch.save(model.state_dict(), best_model_path)

        return avg_test_logloss_bce

def train(worker_id, args, kv):
    print(f"Worker {worker_id} training on {args.device}")
    # Initialize distributed training context.

    kv.begin_setup()
    model = args.model
    torch.manual_seed(args.model_seed)
    model.kv_init(kv, 0, args.opt, worker_id == 0, args.intent_ahead > 0)
    model.to(args.device)
    kv.end_setup()
    kv.wait_sync()

    # full replication
    if args.enforce_full_replication > 0:
        print(f"Enforcing full replication: signal intent for keys 0..{args.num_keys} in time [0, {sys.maxsize}]")
        all_keys = torch.arange(args.num_keys)
        for keys in all_keys.split(2**20):
            kv.intent(keys, 0, sys.maxsize)
            time.sleep(3)
    kv.wait_sync()
    kv.wait_sync()

    best_accuracy = 0
    train_dl, test_dl = ctr.criteo_dataset.init_distributed_dataloaders(rank=worker_id, world_size=args.world_size, batch_size=args.batch_size, num_workers=args.dl_workers, data_root_dir=args.dataset_dir, args=args)

    loss_fn = nn.BCEWithLogitsLoss()

    def intent(batch, device_id):  # function to determine intents.
        if args.intent_ahead > 0:
            # determines intent duration
            target_time = kv.current_clock() + args.intent_ahead
            # calls the intent for each layer
            model.model.wide.wide_linear.intent(batch["sparse_features"].cpu(), target_time)
            model.model.sparse_embedding1.intent(batch["sparse_features"].cpu(), target_time)

        if not batch["sparse_features"].device == torch.device('cpu'):  # no string comparison hopefully
            print(f"WARNING extra memory transfer in intent(). These parts of the batch should be in cpu memory")
        # use the intent to move everything in a non-blocking manner to gpu.
        batch["labels"] = batch["labels"].to(device=device_id, non_blocking=True)  # hardcoded for now.
        batch["dense_features"] = batch["dense_features"].to(device=device_id, non_blocking=True)
        batch["sparse_features"] = batch["sparse_features"].to(device=device_id, non_blocking=True)

    # set the device_id with which the batch is moved to GPU in intent()
    intent_to_device = functools.partial(intent, device_id=args.device)

    kv.barrier() # workers start epoch at the same time

    total_train_time = 0
    start_time = time.time()
    print(f"Setup done, training of first epoch starting now.")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        train_dl.sampler.set_epoch(epoch)  # could modify sample selection, shuffling behavior per epoch
        total_train_loss = 0.0
        total_reg_loss = 0.0
        num_batches = -1

        with tqdm.tqdm(PrefetchIterator(kv, train_dl, args.intent_ahead, intent_to_device), position=worker_id) as tq:
            num_batches = len(tq)
            print(f"Epoch={epoch} actually starting the epoch @ {time.time()-epoch_start}. Number of batches in worker {worker_id}: {num_batches}")
            for batch_id, data in enumerate(tq):
                if batch_id == 0:
                    print(f" Epoch={epoch}, first batch happening {time.time() - epoch_start} after 'Setup done'")

                model.pull()
                labels = data['labels']

                pred = model(data["dense_features"], data["sparse_features"])

                main_loss = loss_fn(input=pred, target=labels)
                reg_loss = model.model.get_regularization(args.l2_reg_linear, args.l2_reg_embedding)
                loss = main_loss + reg_loss

                loss.backward()
                # optimizer step is done in the grad_hooks of the backward pass.
                total_train_loss += loss.item()
                total_reg_loss += reg_loss.item()

                tq.set_postfix({'loss': '%.04f' % loss.item(), 'reg_loss': '%.04f' % reg_loss.item()}, refresh=False)

                kv.advance_clock()

        # synchronize replicas
        print(f"worker {worker_id} finished epoch {epoch+1} in {time.time()-epoch_start}. Avg train loss: {total_train_loss/num_batches}. Avg reg loss: {total_reg_loss/num_batches}", flush=True)
        kv.wait_sync()
        kv.barrier()
        kv.wait_sync()
        epoch_stop = time.time()
        total_train_time += epoch_stop - epoch_start

        # Evaluate on only the first worker/device/GPU only.
        if args.eval_model and worker_id == 0:

            model.eval()
            test_loss = run_eval(model, test_dl, args, epoch)

            print(f"All workers finished epoch {epoch+1} (epoch: {epoch_stop-epoch_start:.3f}s, total: {total_train_time:.3f}s). Test loss: {test_loss}", flush=True)

        kv.barrier()

        # maximum time
        if (args.max_runtime != 0 and
                (total_train_time > args.max_runtime or
                 total_train_time + (epoch_stop-epoch_start) > args.max_runtime * 1.05)):
            print(f"Worker {worker_id} stops after epoch {epoch+1} because max. time is reached: {total_train_time}s (+ 1 epoch) > {args.max_runtime}s", flush=True)
            break

    end_time = time.time()
    print(f"All {epoch} epochs done in {end_time-start_time}s")
    kv.barrier()


if __name__ == "__main__":
    args = parse_arguments()

    # read environment variables when running with tracker
    if args.tracker:
        args = common.process_setup.configure_tracker_launch(args)

    criteo_model = ctr.setup.setup_model(args.feature_dim, args.embedding_dim, args.output_dim, "cpu", args.model_str, True)

    # the model
    args.model = PSDense(criteo_model)

    # setup optimizer and model
    args.opt = PSAdagrad(
        lr=args.learning_rate,
        initial_accumulator_value=args.initial_accumulator_value,
        eps=args.epsilon,
    )

    # calculate parameter lens
    args.lens = args.model.lens()
    args.num_keys = len(args.lens)

    # start the necessary processes
    common.process_setup.start_processes(worker_func=run_worker, args=args)
