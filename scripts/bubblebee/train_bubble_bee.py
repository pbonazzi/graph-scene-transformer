def main(args):

    root : str = os.path.join(OUTPUT_DIR, "bubblebee", "results", args.output)
    exists : bool = os.path.isdir(root)
    os.makedirs(root, exist_ok=True)

    if exists and args.load:
        params = load_config(os.path.join(root, "config.yaml"))
    else:
        params = load_config(args.config)
        params["dataset"] = args.dataset
        params['root'] = root
        params['total_param'] = total_param(params)
        print(f"{bcolors.OKGREEN} Configuration file found. {bcolors.ENDC}")

    device : torch.device = select_device()
    params['device'] = device

    if params["encoder"]["floorplan"] != 'none' and args.dataset == "3dfront":
        raise NotImplementedError(f"{bcolors.FAIL}[E] Floor Plan !{bcolors.ENDC}")
    if params["encoder"]['hidden_dimension'] % params["encoder"]['n_of_attention_heads'] != 0 or \
            params["decoder"]['hidden_dimension'] % params["decoder"]['n_of_attention_heads'] != 0:
        raise ValueError(f"{bcolors.FAIL}[E]  hidden dimensions % number of heads != 0 {bcolors.ENDC}")


    # import dataset
    with open(os.path.join(OUTPUT_DIR, params["dataset"], "processed_data", params["dataset"]+"_train.pkl"), 'rb') as f:
        train_set = pickle.load(f)

    with open(os.path.join(OUTPUT_DIR, params["dataset"], "processed_data", params["dataset"]+"_val.pkl"), 'rb') as f:
        val_set = pickle.load(f)

    train_loader = GraphDataLoader(
        train_set, batch_size=params['data']["batch_size"], shuffle=True, drop_last=False, num_workers=4)

    val_loader = GraphDataLoader(
        val_set, batch_size=params['data']["batch_size"], shuffle=True, drop_last=False, num_workers=4)
        
    # bootstrapping
    t0 = time.time()
    per_epoch_time = []

    #  directories
    log_dir = os.path.join(root, "logs")
    check_dir = os.path.join(root, "checkpoints")
    results_txt = os.path.join(root, "results")
    config_name = os.path.join(root, "config.yaml")
    print(f"{bcolors.OKGREEN}Results saving at {root}{bcolors.ENDC}")

    # deterministic / non deterministic
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    device = params['device']
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    # initialize the model
    model = BubbleBee(params)
    model = model.to(device)

    # load the model
    load = True
    if args.load:
        files = glob.glob(check_dir + '/*.pkl')
        if len(files) != 0:
            epoch_nb_list = []
            for file in files:
                epoch_nb = file.split('_')[-1]
                epoch_nb = int(epoch_nb.split('.')[0])  # extract the number
                epoch_nb_list.append(epoch_nb)
            epoch_nb = max(epoch_nb_list)
            latest_epoch = epoch_nb_list.index(epoch_nb)

            model.load_state_dict(torch.load(
                files[latest_epoch], map_location=torch.device(device)))
            print(f"{bcolors.OKGREEN} Loaded checkpoint{bcolors.ENDC}",
                  ":", epoch_nb)
        else:
            print(
                f"{bcolors.WARNING} No checkpoint was found under this directory path {bcolors.ENDC}")
            load = False

    if not args.load or not load:
        epoch_nb = -1
        check_dir = os.path.join(root, "checkpoints")
        os.makedirs(check_dir, exist_ok=True)
        saved_params = {}
        for key in params.keys():
            saved_params[key] = params[key]
            if saved_params[key].__class__.__name__ in ["device", "int64"]:
                saved_params[key] = str(params[key])
        with open(config_name, "w") as f:
            yaml.dump(saved_params, f, sort_keys=True, indent=4)
        print(f"{bcolors.OKGREEN} Network hyper-parameters saved. {bcolors.ENDC}")

    # tensorboard logging
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # TODO select the scene to display on tensorboard
    # download the scan and write it in tb
    # if params["render"]["download_scan"]:
    #     mesh = get_texturemesh(
    #         scan_id=params["render"]["tensorboard_specific_scene"][1])
    #     try:
    #         writer.add_3d("scan", to_dict_batch([mesh]), step=0)
    #     except ValueError as e:
    #         print('mesh', mesh)
    #         raise e


    # initialize the optimizer
    learning_rate = params['optimizer']
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate['init_lr'],
        betas=learning_rate['betas'],
        weight_decay=learning_rate['weight_decay'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=params["scheduler"]['lr_reduce_factor'],
                                                        patience=params["scheduler"]['lr_schedule_patience'],
                                                        verbose=True)

    print(f"{bcolors.OKGREEN} Optimizer initialized. {bcolors.ENDC}")

    # begin training and evaluation loop
    last_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    try:
        with tqdm(range(params['epochs']), position=0, leave=False) as t:
            for epoch in t:
                start = time.time()
                if epoch < epoch_nb:
                    t.set_description('Synchronizing tqdm %d' % epoch)
                    t.set_postfix(time=time.time() - start)
                    continue

                # anomaly detection
                torch.autograd.set_detect_anomaly(True)
                # train and evaluate
                epoch_train_loss, optimizer = train_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                          epoch=epoch, params=params, writer=writer)

                # get evaluation metrics for val data
                epoch_val_loss, epoch_val_loss_deconstructed, loss_weights \
                    = evaluate_network(model=model, data_loader=val_loader, params=params)

                # early stopping (only in the 2nd half of the training process)
                if epoch_val_loss < last_val_loss:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                last_val_loss = epoch_val_loss
                if epochs_no_improve == params['n_epochs_stop'] and epoch > params['epochs'] / 2:
                    print(
                        f"{bcolors.OKGREEN} Early stopping on epoch {epoch}{bcolors.ENDC}")
                    early_stop = True

                # learning rate
                writer.add_hparams(hparam_dict={"epoch": epoch}, metric_dict={
                    'learning_rate': optimizer.param_groups[0]['lr']})

                writer.add_scalar('train/total_loss', epoch_train_loss, epoch)

                # validation
                writer.add_scalar('val/total_loss', epoch_val_loss, epoch)
                writer.add_scalar(os.path.join(
                    "val", "orientation"), epoch_val_loss_deconstructed['ori'], epoch)
                writer.add_scalar(os.path.join("val", "dimension"),
                                  epoch_val_loss_deconstructed['dim'], epoch)
                writer.add_scalar(os.path.join("val", "location"),
                                  epoch_val_loss_deconstructed['loc'], epoch)
                writer.add_scalar(os.path.join("val", "KL_loss"),
                                  epoch_val_loss_deconstructed['kl'], epoch)

                if params['decoder']["output_object_code"]:
                    writer.add_scalar(os.path.join(
                        "val", "category"), epoch_val_loss_deconstructed['cat'], epoch)

                writer.add_scalar(os.path.join("val", "orientation loss sigma"),
                                  loss_weights[0], epoch)
                writer.add_scalar(os.path.join("val", "dimension loss sigma"),
                                  loss_weights[1], epoch)
                writer.add_scalar(os.path.join("val", "location loss sigma"),
                                  loss_weights[2], epoch)
                writer.add_scalar(os.path.join("val", "KL loss weight"),
                                  loss_weights[3], epoch)
                if params["decoder"]["output_object_code"]:
                    writer.add_scalar(os.path.join(
                        "val", "category loss sigma"), loss_weights[4], epoch)

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(os.path.join(
                            *name.split(".")), values=param.data, global_step=epoch)

                # console output
                t.set_description('Epoch %d' % epoch)
                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss)
                per_epoch_time.append(time.time() - start)

                # saving checkpoints and evaluation metrics
                if epoch == params['epochs'] - 1 or early_stop:
                    torch.save(model.state_dict(), '{}.pkl'.format(
                        check_dir + "/epoch_" + str(epoch)))

                    # get evaluation metrics for training data
                    epoch_train_metrics = compute_eval_metrics(
                        model=model, data_loader=train_loader, params=params, Normalizer=train_set.N)
                    # edge accuracy
                    for key in epoch_train_metrics[0].keys():
                        writer.add_scalar(os.path.join(
                            "evaluation_train", key), epoch_train_metrics[0][key], epoch)
                    # wassersein distance
                    writer.add_scalar(
                        'evaluation_train/wassersein_distance', epoch_train_metrics[1], epoch)
                    # intersection of boxes
                    writer.add_scalar(
                        'evaluation_train/intersection_of_boxes', epoch_train_metrics[2], epoch)

                    # get evaluation metrics for val data
                    epoch_val_metrics = compute_eval_metrics(
                        model=model, data_loader=val_loader, params=params, Normalizer=val_set.N)
                    # edge accuracy
                    for key in epoch_val_metrics[0].keys():
                        writer.add_scalar(os.path.join(
                            "evaluation_val", key), epoch_val_metrics[0][key], epoch)
                    # wassersein distance
                    writer.add_scalar(
                        'evaluation_val/wassersein_distance', epoch_val_metrics[1], epoch)
                    # intersection of boxes
                    writer.add_scalar(
                        'evaluation_val/intersection_of_boxes', epoch_val_metrics[2], epoch)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params["optimizer"]['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                if early_stop:
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training, evaluation loop...')

    print(f"{bcolors.OKGREEN} Computing statistics for posterior. {bcolors.ENDC}")
    train_stats_out = os.path.join(root, "train_stats.npz")
    mean_node_est, cov_node_est, mean_edge_est, cov_edge_est = model.collect_train_statistics(train_loader,
                                                                                              plot=True)

    np.savez(train_stats_out, epoch_nb=epoch, mean_node_est=mean_node_est,
             cov_node_est=cov_node_est, mean_edge_est=mean_edge_est, cov_edge_est=cov_edge_est)

    print(f"{bcolors.OKGREEN} Saving the results, please wait. {bcolors.ENDC}")

    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    print(f"{bcolors.OKGREEN}Results saved at {root}{bcolors.ENDC}")
    writer.close()

    # write the results in out_dir/results folder
    with open(results_txt + '.txt', 'w') as f:
        f.write("""Dataset: {},\n\nparams={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTest Loss: {:.4f}\nTrain Loss: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""
                .format(dataset_name, params, model, params['total_param'],
                        epoch_val_loss, epoch_train_loss, epoch, (time.time() - t0) / 3600, np.mean(per_epoch_time)))

    return





if __name__ == "__main__":
    
    import warnings, random, glob, os, sys, inspect, pdb, time 
    import argparse
    import json, pickle, yaml
    from tqdm import tqdm

    import numpy as np
    import open3d as o3d
    from open3d.visualization.tensorboard_plugin import summary
    from open3d.visualization.tensorboard_plugin.util import to_dict_batch

    import torch
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    from dgl.dataloading import GraphDataLoader

    # custom 

    currentdir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    sys.path.insert(0, parentdir)

    from config.paths import OUTPUT_DIR
    from render.colors import bcolors
    
    from scripts.preprocess_dgl import Dataset3DSSG # pickled dataset class
    from scripts import  select_device, total_param, load_config
    from scripts.bubblebee.utils.train import train_epoch
    from scripts.bubblebee.utils.eval import evaluate_network, compute_eval_metrics
    
    from network.models.model import BubbleBee
    from network.models.model_3dssg import Model3DSSG
    

    # user

    parser = argparse.ArgumentParser(
        description='I solemnly swear that I am up to no good.')

    parser.add_argument('--config', '--c',
                        help="Path to configuration file, see config/config.json")
    parser.add_argument('--dataset', '--d', required=True,
                        help='dataset name')
    parser.add_argument('--output', '--o', required=True,
                        help='dir name for loading or saving checkpoints')
    parser.add_argument('--warnings', default=False,
                        help='library warnings')
    parser.add_argument('--load', default=False,
                        help='continue previous run')
    args = parser.parse_args()

    if args.warnings:
        warnings.filterwarnings("ignore")
        o3d.utility.set_verbosity_level((o3d.utility.VerbosityLevel(0)))


    # parallel

    os.environ["OMP_NUM_THREADS"] = "10"  
    os.environ["OPENBLAS_NUM_THREADS"] = "10"
    os.environ["MKL_NUM_THREADS"] = "10" 
    os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
    os.environ["NUMEXPR_NUM_THREADS"] = "10"
    torch.set_num_threads(10)
    torch.set_num_interop_threads(10)


    main(args)
