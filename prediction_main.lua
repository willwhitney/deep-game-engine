
require 'sys'
require 'xlua'
require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'optim'

require 'modules/KLDCriterion'
require 'modules/LinearCR'
require 'modules/Reparametrize'
require 'modules/SelectiveOutputClamp'
require 'modules/SelectiveGradientFilter'

require 'rmsprop'
require 'testf'
require 'utils'
require 'networks'
require 'lfs'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Train a network to store particular information in particular nodes.')
cmd:text()
cmd:text('Options')

cmd:text('Change these options:')
cmd:option('--import',            '',             'the containing folder of the network to load in. does nothing with `no_load`')
cmd:option('--networks_dir',      'networks',     'the directory to save the resulting networks in')
cmd:option('--name',              'default',      'the name for this network. used for saving the network and results')
cmd:option('--datasetdir',        'dataset',      'dataset source directory')

cmd:option('--dim_hidden',        200,            'dimension of the representation layer')
cmd:option('--dim_prediction',    512,            'dimension of the prediction layer')
cmd:option('--feature_maps',      96,             'number of feature maps')

cmd:option('--learning_rate',     -0.0005,        'learning rate for the network')
cmd:option('--momentum_decay',    0.1,            'decay rate for momentum in rmsprop')
cmd:option('--update_decay',      0.01,           'update decay rate')

cmd:text()
cmd:text()

cmd:text("Probably don't change these:")
cmd:option('--threads', 1, 'how many threads to use in torch')
cmd:option('--num_train_batches', 24999,'number of batches to train with per epoch')
cmd:option('--num_test_batches', 3999, 'number of batches to test with')
cmd:option('--epoch_size', 5000, 'number of batches to test with')
cmd:option('--tests_per_epoch', 200, 'number of test batches to run every epoch')
cmd:option('--bsize', 30, 'number of samples per batch_images')

cmd:text()

opt = cmd:parse(arg)
opt.save = paths.concat(opt.networks_dir, opt.name)
os.execute('mkdir -p "' .. opt.save .. '"')

torch.setnumthreads(opt.threads)

-- log out the options used for creating this network to a file in the save directory.
-- super useful when you're moving folders around so you don't lose track of things.
local f = assert(io.open(opt.save .. '/cmd_options.txt', 'w'))
for key, val in pairs(opt) do
  f:write(tostring(key) .. ": " .. tostring(val) .. "\n")
end
f:flush()
f:close()

MODE_TRAINING = "train"
MODE_TEST = "test"


model = build_atari_prediction_network(opt.dim_hidden, opt.feature_maps, opt.dim_prediction)


criterion = nn.BCECriterion()
criterion.sizeAverage = false

KLD = nn.KLDCriterion()
KLD.sizeAverage = false

criterion:cuda()
KLD:cuda()
model:cuda()
cutorch.synchronize()

parameters, gradients = model:getParameters()
print('Num parameters before loading:', #parameters)

if opt.import ~= '' then
  -- load all the values from the network stored in opt.import
  lowerboundlist = torch.load(paths.concat(opt.networks_dir, opt.import, 'lowerbound.t7'))
  lowerbound_test_list = torch.load(paths.concat(opt.networks_dir, opt.import, 'lowerbound_test.t7'))
  state = torch.load(paths.concat(opt.networks_dir, opt.import, 'state.t7'))
  p = torch.load(paths.concat(opt.networks_dir, opt.import, 'parameters.t7'))
  print('Loaded parameter size:', #p)
  parameters:copy(p)
  epoch = lowerboundlist:size(1)
else
  epoch = 0
end

testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
reconstruction = 0

while true do
  epoch = epoch + 1
  local lowerbound = 0
  local time = sys.clock()

  for i = 1, opt.epoch_size do
    xlua.progress(i, opt.epoch_size)

    --Prepare Batch
    local batch_images, batch_actions = load_random_atari_full_batch(MODE_TRAINING)
    batch_images = batch_images:cuda()
    batch_actions = batch_actions:cuda()

    local input_images = batch_images[{{1, batch_images:size(1) - 1}}]
    local input_actions = batch_actions[{{1, batch_images:size(1) - 1}}]
    input_actions:resize(batch_images:size(1) - 1, 1)
    local target = batch_images[{{2, batch_images:size(1)}}]

    --Optimization function
    local opfunc = function(x)
      collectgarbage()

      if x ~= parameters then
        parameters:copy(x)
      end

      model:zeroGradParameters()
      local f = model:forward({input_images, input_actions})
      -- print(f:size())

      -- local target = target or batch_images.new()
      -- target:resizeAs(f):copy(batch_images)

      local err = - criterion:forward(f, target)
      local df_dw = criterion:backward(f, target):mul(-1)

      model:backward(input_images, df_dw)
      local predictor_output = predictor.output

      local KLDerr = KLD:forward(predictor_output, target)
      local dKLD_dw = KLD:backward(predictor_output, target)

      predictor:backward(z_in.output, dKLD_dw)
      -- print(predictor.gradInput[1]:size())
      encoder:backward(input_images, predictor.gradInput[1])

      local lowerbound = err  + KLDerr

      if opt.verbose then
        print("BCE",err/ (batch_images:size(1) - 1))
        print("KLD", KLDerr/(batch_images:size(1) - 1))
        print("lowerbound", lowerbound/(batch_images:size(1) - 1))
      end

      return lowerbound, gradients
    end -- /opfunc

    x, batchlowerbound = rmsprop(opfunc, parameters, config, state)

    lowerbound = lowerbound + batchlowerbound[1]
  end

  print("\nEpoch: " .. epoch ..
    " Lowerbound: " .. lowerbound/opt.num_train_batches ..
    " time: " .. sys.clock() - time)

  --Keep track of the lowerbound over time
  if lowerboundlist then
    lowerboundlist = torch.cat(lowerboundlist
      ,torch.Tensor(1,1):fill(lowerbound/opt.num_train_batches)
      ,1)
  else
    lowerboundlist = torch.Tensor(1,1):fill(lowerbound/opt.num_train_batches)
  end


  -- save the current net
  if true then
    local filename = paths.concat(opt.save, 'vxnet.net')
    os.execute('mkdir -p "' .. sys.dirname(filename) ..'"')
    if paths.filep(filename) then
      os.execute('mv "' .. filename .. '" "' .. filename .. '.old"')
    end

    print('<trainer> saving network to '..filename)
    torch.save(filename, model)
  end


  -- Compute the lowerbound of the test set and save it
  lowerbound_test = test_atari_prediction(false)
  if true then
    if lowerbound_test_list then
      lowerbound_test_list = torch.cat(lowerbound_test_list
                                      ,torch.Tensor(1,1):fill(lowerbound_test/opt.num_test_batches)
                                      ,1)
    else
      lowerbound_test_list = torch.Tensor(1,1):fill(lowerbound_test/opt.num_test_batches)
    end

    print('testlowerbound = ' .. lowerbound_test/opt.num_test_batches)

    --Save everything to be able to restart later
    torch.save(opt.save .. '/parameters.t7', parameters)
    torch.save(opt.save .. '/state.t7', state)
    torch.save(opt.save .. '/lowerbound.t7', torch.Tensor(lowerboundlist))
    torch.save(opt.save .. '/lowerbound_test.t7', torch.Tensor(lowerbound_test_list))
  end

  -- plot errors
  if false then
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    testLogger:plot()
  end
end

