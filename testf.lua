
-- test function for reconstruction task
function test_atari_reconstruction(saveAll)
  -- in case it didn't already exist
  os.execute('mkdir -p "' .. 'tmp' .. '"')

  -- local vars
  local time = sys.clock()
  -- test over given dataset
  print('<trainer> on testing Set:')
  local reconstruction = 0
  local lowerbound = 0

  local save_dir = paths.concat('tmp', opt.name, 'epoch_' .. epoch)
  os.execute('mkdir -p "' .. save_dir .. '"')

   for test_index = 1, opt.tests_per_epoch do
      collectgarbage()
      -- create mini batch
      local raw_inputs = load_random_atari_images_batch(MODE_TEST, test_index)
      local targets = raw_inputs

      inputs = raw_inputs:cuda()
      -- disp progress
      xlua.progress(test_index, opt.tests_per_epoch)

      -- test samples
      local preds = model:forward(inputs)

      local f = preds
      local target = targets
      local err = - criterion:forward(f, target:cuda())
      local encoder_output = model:get(1).output
      local KLDerr = KLD:forward(encoder_output, target)
      lowerbound = lowerbound + err + KLDerr


      preds = preds:float()

      reconstruction = reconstruction + torch.sum(torch.pow(preds-targets,2))

      if saveAll then
        torch.save(save_dir..'/preds' .. test_index, preds)
        torch.save(save_dir..'/truth' .. test_index, targets)
      else
        if test_index < 10 then
            torch.save(save_dir..'/preds' .. test_index, preds)
            torch.save(save_dir..'/truth' .. test_index, targets)
        end
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / opt.tests_per_epoch
   print("<trainer> time to test 1 sample = " .. (time * 1000) .. 'ms')

   -- print confusion matrix
   reconstruction = reconstruction / (opt.bsize * opt.tests_per_epoch * 3 * 150 * 150)
   print('mean MSE error (test set)', reconstruction)
   testLogger:add{['% mean class accuracy (test set)'] = reconstruction}
   reconstruction = 0
   return lowerbound
end

-- test function for prediction task
function test_atari_prediction(saveAll)
  -- in case it didn't already exist
  os.execute('mkdir -p "' .. 'tmp' .. '"')

  -- local vars
  local time = sys.clock()
  -- test over given dataset
  print('<trainer> on testing Set:')
  local reconstruction = 0
  local lowerbound = 0

  local save_dir = paths.concat('tmp', opt.name, 'epoch_' .. epoch)
  os.execute('mkdir -p "' .. save_dir .. '"')

   for test_index = 1, opt.tests_per_epoch do
      collectgarbage()
      -- create mini batch
      local batch_images, batch_actions = load_random_atari_full_batch(MODE_TRAINING)
      batch_images = batch_images:cuda()
      batch_actions = batch_actions:cuda()

      local input_images = batch_images[{{1, batch_images:size(1) - 1}}]
      local input_actions = batch_actions[{{1, batch_images:size(1) - 1}}]
      input_actions:resize(batch_images:size(1) - 1, 1)
      local target = batch_images[{{2, batch_images:size(1)}}]

      -- disp progress
      xlua.progress(test_index, opt.tests_per_epoch)

      -- test samples
      local preds = model:forward({input_images, input_actions})

      local f = preds
      local err = - criterion:forward(f, target)
      local KLDerr = KLD:forward(predictor.output, target)
      lowerbound = lowerbound + err + KLDerr

      preds = preds:float()

      reconstruction = reconstruction + torch.sum(torch.pow(preds-target:float(),2))

      if saveAll then
        torch.save(save_dir..'/preds' .. test_index, preds)
        torch.save(save_dir..'/truth' .. test_index, target:float())
      else
        if test_index < 10 then
          torch.save(save_dir..'/preds' .. test_index, preds)
          torch.save(save_dir..'/truth' .. test_index, target:float())
        end
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / opt.tests_per_epoch
   print("<trainer> time to test 1 sample = " .. (time * 1000) .. 'ms')

   -- print confusion matrix
   reconstruction = reconstruction / (opt.bsize * opt.tests_per_epoch * 3 * 150 * 150)
   print('mean MSE error (test set)', reconstruction)
   testLogger:add{['% mean class accuracy (test set)'] = reconstruction}
   reconstruction = 0
   return lowerbound
end
