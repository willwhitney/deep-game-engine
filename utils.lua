
function load_atari_images_batch(mode, id)
    local b = torch.load(paths.concat(opt.datasetdir, mode, 'images_batch_' .. id))
    if opt.grayscale then
        return grayscale(b)
    else
        return b
    end
end

function load_atari_actions_batch(mode, id)
    return torch.load(paths.concat(opt.datasetdir, mode, 'actions_batch_' .. id))
end

function load_atari_full_batch(mode, id)
    return  load_atari_images_batch(mode, id),
            load_atari_actions_batch(mode, id)
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

function load_random_atari_full_batch(mode)
    local batch_id = nil
    if mode == MODE_TRAINING then
        batch_id = torch.random(opt.num_train_batches)
    elseif mode == MODE_TEST then
        batch_id = torch.random(opt.num_test_batches)
    else
        assert(false)
    end

    return load_atari_full_batch(mode, batch_id)
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

function grayscale_image(img)
    return img[1] * 0.33 + img[2] * 0.33 + img[3] * 0.33
end

function grayscale(image_batch)
    local grayscale_batch = torch.Tensor(image_batch:size(1),
                                         1,
                                         image_batch:size(3),
                                         image_batch:size(4))
    for i = 1, image_batch:size(1) do
        grayscale_batch[i][1] = grayscale_image(image_batch[i])
    end
    return grayscale_batch
end
