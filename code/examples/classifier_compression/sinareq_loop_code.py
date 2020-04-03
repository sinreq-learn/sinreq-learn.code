# start 
def train(train_loader, model, criterion, optimizer, epoch,
          compression_scheduler, loggers, args):
   """Training loop for one epoch."""
    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    # For Early Exit, we define statistics for each exit
    # So exiterrors is analogous to classerr for the non-Early Exit case
    if args.earlyexit_lossweights:
        args.exiterrors = []
        for exitnum in range(args.num_exits):
            args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)
    epoch_frac = args.partial_epoch
    steps_per_frac_epoch = math.ceil((total_samples*epoch_frac) / batch_size)

    # Switch to train mode
    model.train()
    end = time.time()

    for train_step, (inputs, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.add(time.time() - end)
        inputs, target = inputs.to('cuda'), target.to('cuda')

        if train_step == steps_per_frac_epoch:
            break
        # Execute the forward phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

        if args.kd_policy is None:
            output = model(inputs)
        else:
            output = args.kd_policy.forward(inputs)
        if not args.earlyexit_lossweights:

            sinareq_loss = 0
            bitwidth_loss = 0
            layers = list()

            last_epoch = 999
            if (train_step == last_epoch):

            for each in  ['state_dict'].keys():
               if "weight" in each: layers.append(each)

            for layer_name in layers:
               step = 1/(2**(model.module[layer_name].bitwidth)-1) 
               kernel = model.module[layer_name].weight
               """ 1) Weight quantization regularization """
               sinareq_loss += torch.mean(torch.pow(torch.sin(pi*(kernel+shift)/step),power)/2**(model.module[layer_name].bitwidth)) 
               """ 2) Bitwidth regularization (learning the bitwidth (the sinusoidal period)) """
               bitwidth_loss += model.module[layer_name].bitwidth

            loss = criterion(output, target) + (lambda_w * sinareq_loss) + (lambda_b * bitwidth_loss)

            # Measure accuracy and record loss
            classerr.add(output.data, target)
        else:
            # Measure accuracy and record loss
            loss = earlyexit_loss(output, target, criterion, args)
        losses[OBJECTIVE_LOSS_KEY].add(loss.item())

        if compression_scheduler:
            # Before running the backward phase, we allow the scheduler to modify the loss
            # (e.g. add regularization loss)
            agg_loss = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss,
                                                                  optimizer=optimizer, return_loss_components=True)
            loss = agg_loss.overall_loss
            losses[OVERALL_LOSS_KEY].add(loss.item())
            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                losses[lc.name].add(lc.value.item())

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)

        # measure elapsed time
        batch_time.add(time.time() - end)
        steps_completed = (train_step+1)

        if steps_completed % args.print_freq == 0:
            # Log some statistics
            errs = OrderedDict()
            if not args.earlyexit_lossweights:
                errs['Top1'] = classerr.value(1)
                errs['Top5'] = classerr.value(5)
            else:
                # for Early Exit case, the Top1 and Top5 stats are computed for each exit.
                for exitnum in range(args.num_exits):
                    errs['Top1_exit' + str(exitnum)] = args.exiterrors[exitnum].value(1)
                    errs['Top5_exit' + str(exitnum)] = args.exiterrors[exitnum].value(5)

            stats_dict = OrderedDict()
            for loss_name, meter in losses.items():
                stats_dict[loss_name] = meter.mean
            stats_dict.update(errs)
            stats_dict['LR'] = optimizer.param_groups[0]['lr']
            stats_dict['Time'] = batch_time.mean
            stats = ('Peformance/Training/', stats_dict)

            params = model.named_parameters() if args.log_params_histograms else None
            distiller.log_training_progress(stats,
                                            params,
                                            epoch, steps_completed,
                                            steps_per_epoch, args.print_freq,
                                            loggers)
        end = time.time()
