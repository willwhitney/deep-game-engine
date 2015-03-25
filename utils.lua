
function load_atari_images_batch(mode, id)
    return torch.load(paths.concat(opt.datasetdir, mode, 'images_batch_' .. id))
end

function load_random_atari_images_batch(mode)
    local batch_id = nil
    if mode == MODE_TRAINING then
        batch_id = torch.random(opt.num_train_batches)
    elseif mode == MODE_TEST then
        batch_id = torch.random(opt.num_test_batches)
    else
        assert(false)
    end

    return load_atari_images_batch(mode, batch_id)
end

function getLowerbound(data)
    local lowerbound = 0
    N_data = num_test_batches
    for i = 1, N_data, batchSize do
        local batch = data[{{i,i+batchSize-1},{}}]
        local f = model:forward(batch)
        local target = target or batch.new()
        target:resizeAs(f):copy(batch)
        local err = - criterion:forward(f, target)

        local encoder_output = model:get(1).output

        local KLDerr = KLD:forward(encoder_output, target)

        lowerbound = lowerbound + err + KLDerr
    end
    return lowerbound
end
